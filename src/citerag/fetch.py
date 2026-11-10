"""Download a document from a URL without letting the server be used to reach internal hosts.

Fetching a URL on behalf of a caller is a classic SSRF risk: `http://169.254.169.254/` or a
hostname that resolves to `10.0.0.5` would let anyone who can reach the API read internal
services. `Fetcher` therefore resolves every host itself, refuses non-public addresses, and then
connects to the exact address it checked (pinning it, so DNS can't answer differently a moment
later). Redirects are followed by hand so each hop goes through the same check.
"""

import ipaddress
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath

import httpx

from citerag.loaders import SUPPORTED_SUFFIXES, UnsupportedFileError

# (host, port) -> IP addresses as strings, like socket.getaddrinfo returns them.
Resolver = Callable[[str, int], list[str]]

DEFAULT_MAX_BYTES = 25 * 1024 * 1024
DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_REDIRECTS = 5
USER_AGENT = "citerag (+https://github.com/GuiDobelin/citerag)"

_CONTENT_TYPES = {
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/x-markdown": ".md",
}
# Types that say nothing about the format, so the URL path suffix decides.
_GENERIC_TYPES = {"", "application/octet-stream", "binary/octet-stream"}
_TEXT_SUFFIXES = {".html", ".htm", ".txt", ".md"}
_REDIRECTS = {301, 302, 303, 307, 308}
# NAT64 addresses embed an IPv4 address in their last 32 bits.
_NAT64 = ipaddress.ip_network("64:ff9b::/96")


class URLFetchError(ValueError):
    """The URL could not be downloaded (network error, timeout, or an error status)."""


class BlockedURLError(URLFetchError):
    """The URL is invalid or points to an address the fetcher is not allowed to reach."""


class ResponseTooLargeError(URLFetchError):
    """The response body is larger than the configured limit."""


@dataclass(frozen=True)
class FetchedDocument:
    url: str
    """The final URL, after redirects."""
    filename: str
    """A name whose suffix tells `citerag.loaders` how to read `data`."""
    data: bytes


def system_resolver(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as exc:
        raise URLFetchError(f"Could not resolve host '{host}'.") from exc
    return [str(info[4][0]) for info in infos]


def is_public_address(ip: str) -> bool:
    """True if `ip` is a globally routable unicast address (IPv4 or IPv6)."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address):
        # Addresses that carry an IPv4 address are judged by that address.
        embedded = addr.ipv4_mapped or addr.sixtofour
        if embedded is None and addr.teredo:
            embedded = addr.teredo[1]
        if embedded is None and addr in _NAT64:
            embedded = ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)
        if embedded is not None:
            return is_public_address(str(embedded))
    return addr.is_global and not (addr.is_multicast or addr.is_reserved)


def filename_for(url: httpx.URL, content_type: str) -> str:
    """Pick a loader from the Content-Type, falling back to the URL path suffix."""
    media_type = content_type.split(";", 1)[0].strip().lower()
    suffix = PurePosixPath(url.path).suffix.lower()
    if media_type in _CONTENT_TYPES:
        suffix = _CONTENT_TYPES[media_type]
    elif media_type not in _GENERIC_TYPES or suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileError(
            f"Unsupported content type '{media_type or 'unknown'}' at {url}. "
            f"Supported: HTML, PDF, DOCX, plain text and Markdown."
        )
    stem = PurePosixPath(url.path).stem or url.host or "document"
    return f"{stem}{suffix}"


def _ascii_host(url: httpx.URL) -> str:
    """The host in ASCII (IDNA) form, without the brackets around IPv6 addresses."""
    return url.raw_host.decode("ascii").strip("[]")


def _size(n: int) -> str:
    return f"{n / (1024 * 1024):g} MB" if n >= 1024 * 1024 else f"{n} bytes"


def _to_utf8(data: bytes, content_type: str) -> bytes:
    """Re-encode text declared in another charset, since the loaders read text as UTF-8."""
    params = content_type.split(";")[1:]
    charset = next(
        (
            v.strip().strip("\"'")
            for k, _, v in (p.partition("=") for p in params)
            if k.strip().lower() == "charset"
        ),
        None,
    )
    if not charset or charset.lower().replace("_", "-") in {"utf-8", "utf8"}:
        return data
    try:
        return data.decode(charset, errors="replace").encode("utf-8")
    except LookupError:
        return data


class Fetcher:
    """Downloads documents over HTTP(S), refusing non-public addresses unless `allow_private`.

    `resolver` and `transport` exist for tests; by default the system resolver and a real
    network connection are used. Proxy environment variables are ignored on purpose: going
    through a proxy would connect somewhere other than the address that was checked.
    """

    def __init__(
        self,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        allow_private: bool = False,
        timeout: float = DEFAULT_TIMEOUT,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        resolver: Resolver | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1.")
        if timeout <= 0:
            raise ValueError("timeout must be > 0.")
        if max_redirects < 0:
            raise ValueError("max_redirects must be >= 0.")
        self.max_bytes = max_bytes
        self.allow_private = allow_private
        self.timeout = timeout
        self.max_redirects = max_redirects
        self.resolver = resolver or system_resolver
        self.transport = transport

    def fetch(self, url: str) -> FetchedDocument:
        try:
            current = httpx.URL(url.strip())
        except (httpx.InvalidURL, TypeError) as exc:
            raise BlockedURLError(f"Invalid URL: {exc}") from exc
        deadline = time.monotonic() + self.timeout
        with httpx.Client(
            transport=self.transport,
            timeout=self.timeout,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            for _ in range(self.max_redirects + 1):
                response = self._open(client, current)
                try:
                    if response.status_code in _REDIRECTS and "location" in response.headers:
                        current = self._redirect_target(current, response.headers["location"])
                        continue
                    if response.status_code >= 400:
                        raise URLFetchError(f"{current} returned HTTP {response.status_code}.")
                    content_type = response.headers.get("content-type", "")
                    filename = filename_for(current, content_type)
                    data = self._read(response, deadline, current)
                finally:
                    response.close()
                if PurePosixPath(filename).suffix in _TEXT_SUFFIXES:
                    data = _to_utf8(data, content_type)
                return FetchedDocument(url=str(current), filename=filename, data=data)
        raise URLFetchError(f"Too many redirects (more than {self.max_redirects}).")

    def _redirect_target(self, current: httpx.URL, location: str) -> httpx.URL:
        try:
            return current.join(location)
        except httpx.InvalidURL as exc:
            raise URLFetchError(f"Invalid redirect location: {location!r}") from exc

    def _check(self, url: httpx.URL) -> str:
        """Validate `url` and return the IP address to connect to."""
        if url.scheme not in ("http", "https"):
            raise BlockedURLError(f"Only http and https URLs are allowed, not '{url.scheme}'.")
        if not url.host:
            raise BlockedURLError(f"URL has no host: {url}")
        if url.userinfo:
            raise BlockedURLError("URLs with a username or password are not allowed.")
        port = url.port or (443 if url.scheme == "https" else 80)
        addresses = self.resolver(_ascii_host(url), port)
        if not addresses:
            raise URLFetchError(f"Could not resolve host '{url.host}'.")
        if not self.allow_private:
            # Every address must be public, otherwise a hostname with one public and one
            # private record could still reach the private one on a later connection.
            blocked = [a for a in addresses if not is_public_address(a)]
            if blocked:
                raise BlockedURLError(
                    f"'{url.host}' resolves to a non-public address ({blocked[0]}). "
                    "Set CITERAG_ALLOW_PRIVATE_URLS=true to allow internal hosts."
                )
        return addresses[0]

    def _open(self, client: httpx.Client, url: httpx.URL) -> httpx.Response:
        ip = self._check(url)
        # Connect to the checked address, but send the real hostname in the Host header and in
        # TLS SNI so virtual hosts and certificate verification work as usual.
        pinned = url.copy_with(host=ip)
        headers = {
            "Host": url.netloc.decode("ascii"),
            "User-Agent": USER_AGENT,
            "Accept": "text/html, application/pdf, text/plain, text/markdown, */*;q=0.5",
        }
        request = client.build_request(
            "GET", pinned, headers=headers, extensions={"sni_hostname": _ascii_host(url)}
        )
        try:
            return client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise URLFetchError(f"Timed out fetching {url}.") from exc
        except httpx.HTTPError as exc:
            raise URLFetchError(f"Could not fetch {url}: {exc}") from exc

    def _read(self, response: httpx.Response, deadline: float, url: httpx.URL) -> bytes:
        too_large = ResponseTooLargeError(
            f"Response from {url} is larger than the {_size(self.max_bytes)} limit."
        )
        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > self.max_bytes:
            raise too_large
        body = bytearray()
        try:
            # iter_bytes yields decompressed data, so the limit also stops compression bombs.
            for piece in response.iter_bytes():
                body += piece
                if len(body) > self.max_bytes:
                    raise too_large
                if time.monotonic() > deadline:
                    raise URLFetchError(f"Timed out fetching {url}.")
        except httpx.TimeoutException as exc:
            raise URLFetchError(f"Timed out fetching {url}.") from exc
        except httpx.HTTPError as exc:
            raise URLFetchError(f"Error reading the response from {url}: {exc}") from exc
        return bytes(body)
