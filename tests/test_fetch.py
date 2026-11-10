import gzip
import io

import httpx
import pytest
from conftest import LISBON
from pypdf import PdfWriter

from citerag.fetch import (
    BlockedURLError,
    Fetcher,
    ResponseTooLargeError,
    URLFetchError,
    filename_for,
    is_public_address,
)
from citerag.loaders import UnsupportedFileError

PUBLIC_IP = "93.184.215.14"


class FakeDNS:
    """Resolver with fixed answers; unknown hosts resolve to a public address."""

    def __init__(self, records: dict[str, list[str]] | None = None) -> None:
        self.records = records or {}
        self.lookups: list[tuple[str, int]] = []

    def __call__(self, host: str, port: int) -> list[str]:
        self.lookups.append((host, port))
        return self.records.get(host, [PUBLIC_IP])


def make_fetcher(handler, dns=None, **kwargs) -> Fetcher:
    kwargs.setdefault("max_bytes", 1024 * 1024)
    return Fetcher(resolver=dns or FakeDNS(), transport=httpx.MockTransport(handler), **kwargs)


def html_page(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/html; charset=utf-8"},
        text=f"<html><body><p>{LISBON}</p><script>tracking()</script></body></html>",
    )


@pytest.mark.parametrize(
    ("ip", "public"),
    [
        ("93.184.215.14", True),
        ("2606:2800:21f:cb07:6820:80da:af6b:8b2c", True),
        ("127.0.0.1", False),
        ("10.1.2.3", False),
        ("172.16.0.1", False),
        ("192.168.1.1", False),
        ("169.254.169.254", False),  # cloud metadata endpoint
        ("100.64.0.1", False),  # carrier-grade NAT
        ("0.0.0.0", False),
        ("224.0.0.1", False),
        ("240.0.0.1", False),
        ("255.255.255.255", False),
        ("::1", False),
        ("::", False),
        ("fe80::1", False),
        ("fc00::1", False),
        ("ff02::1", False),
        ("::ffff:127.0.0.1", False),  # IPv4-mapped
        ("::ffff:93.184.215.14", True),
        ("2002:7f00:1::", False),  # 6to4 around 127.0.0.1
        ("64:ff9b::a00:1", False),  # NAT64 around 10.0.0.1
        ("not-an-ip", False),
    ],
)
def test_is_public_address(ip, public):
    assert is_public_address(ip) is public


@pytest.mark.parametrize(
    ("url", "content_type", "expected"),
    [
        ("https://x.org/guide", "text/html; charset=utf-8", "guide.html"),
        ("https://x.org/", "application/xhtml+xml", "x.org.html"),
        ("https://x.org/paper", "application/pdf", "paper.pdf"),
        ("https://x.org/notes.txt", "text/plain", "notes.txt"),
        ("https://x.org/README", "text/markdown", "README.md"),
        ("https://x.org/report.docx", "application/octet-stream", "report.docx"),
        ("https://x.org/report.PDF", "", "report.pdf"),
    ],
)
def test_filename_for_prefers_content_type_then_suffix(url, content_type, expected):
    assert filename_for(httpx.URL(url), content_type) == expected


@pytest.mark.parametrize(
    ("url", "content_type"),
    [
        ("https://x.org/photo.png", "image/png"),
        ("https://x.org/page.html", "image/png"),  # a real type wins over the suffix
        ("https://x.org/download", "application/octet-stream"),
    ],
)
def test_filename_for_rejects_unsupported(url, content_type):
    with pytest.raises(UnsupportedFileError):
        filename_for(httpx.URL(url), content_type)


def test_fetch_pins_the_checked_address_and_keeps_the_hostname():
    seen = []

    def handler(request):
        seen.append(request)
        return html_page(request)

    dns = FakeDNS()
    doc = make_fetcher(handler, dns).fetch("https://docs.example.com/guide?lang=en")

    assert doc.url == "https://docs.example.com/guide?lang=en"
    assert doc.filename == "guide.html"
    assert dns.lookups == [("docs.example.com", 443)]
    request = seen[0]
    assert request.url.host == PUBLIC_IP
    assert request.headers["host"] == "docs.example.com"
    assert request.extensions["sni_hostname"] == "docs.example.com"


def test_fetch_idn_host_uses_ascii_form():
    dns = FakeDNS()
    seen = []
    fetcher = make_fetcher(lambda r: seen.append(r) or html_page(r), dns)
    fetcher.fetch("https://bücher.example/")
    assert dns.lookups == [("xn--bcher-kva.example", 443)]
    assert seen[0].headers["host"] == "xn--bcher-kva.example"


@pytest.mark.parametrize(
    "url",
    ["ftp://example.com/file.txt", "file:///etc/passwd", "https://user:pw@example.com/", "nope"],
)
def test_fetch_rejects_invalid_urls(url):
    fetcher = make_fetcher(lambda r: pytest.fail("no request expected"))
    with pytest.raises(BlockedURLError):
        fetcher.fetch(url)


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1/", "http://[::1]:8000/", "http://169.254.169.254/latest/meta-data"]
)
def test_fetch_refuses_private_ip_literals(url):
    dns = FakeDNS(
        {"127.0.0.1": ["127.0.0.1"], "::1": ["::1"], "169.254.169.254": ["169.254.169.254"]}
    )
    fetcher = make_fetcher(lambda r: pytest.fail("no request expected"), dns)
    with pytest.raises(BlockedURLError, match="CITERAG_ALLOW_PRIVATE_URLS"):
        fetcher.fetch(url)


def test_fetch_refuses_host_with_any_private_record():
    dns = FakeDNS({"mixed.example": [PUBLIC_IP, "10.0.0.5"]})
    fetcher = make_fetcher(lambda r: pytest.fail("no request expected"), dns)
    with pytest.raises(BlockedURLError, match="10.0.0.5"):
        fetcher.fetch("https://mixed.example/")


def test_fetch_refuses_redirect_to_private_address():
    dns = FakeDNS({"internal.corp": ["10.0.0.5"]})
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "http://internal.corp/admin"})

    with pytest.raises(BlockedURLError):
        make_fetcher(handler, dns).fetch("https://example.com/go")
    assert len(requests) == 1


def test_fetch_follows_relative_redirect_and_reports_final_url():
    def handler(request):
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/new/page"})
        return html_page(request)

    doc = make_fetcher(handler).fetch("https://example.com/old")
    assert doc.url == "https://example.com/new/page"


def test_fetch_limits_redirects():
    def handler(request):
        return httpx.Response(302, headers={"location": "/again"})

    with pytest.raises(URLFetchError, match="Too many redirects"):
        make_fetcher(handler, max_redirects=2).fetch("https://example.com/")


def test_fetch_allows_private_addresses_when_opted_in():
    dns = FakeDNS({"wiki.internal": ["10.0.0.5"]})
    doc = make_fetcher(html_page, dns, allow_private=True).fetch("http://wiki.internal/home")
    assert doc.url == "http://wiki.internal/home"


def test_fetch_error_status():
    with pytest.raises(URLFetchError, match="HTTP 404"):
        make_fetcher(lambda r: httpx.Response(404)).fetch("https://example.com/missing")


def test_fetch_network_error_is_wrapped():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(URLFetchError, match="Could not fetch"):
        make_fetcher(handler).fetch("https://example.com/")


def test_fetch_rejects_declared_oversized_body():
    def handler(request):
        return httpx.Response(
            200, headers={"content-type": "text/plain", "content-length": "5000"}, content=b"x"
        )

    with pytest.raises(ResponseTooLargeError):
        make_fetcher(handler, max_bytes=1000).fetch("https://example.com/a.txt")


def test_fetch_stops_reading_oversized_stream():
    produced = []

    class Endless(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(1000):
                produced.append(1)
                yield b"x" * 100

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/plain"}, stream=Endless())

    with pytest.raises(ResponseTooLargeError):
        make_fetcher(handler, max_bytes=1000).fetch("https://example.com/a.txt")
    assert len(produced) <= 11


def test_fetch_limit_applies_to_decompressed_size():
    bomb = gzip.compress(b"a" * 100_000)

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/plain", "content-encoding": "gzip"},
            content=bomb,
        )

    assert len(bomb) < 1000
    with pytest.raises(ResponseTooLargeError):
        make_fetcher(handler, max_bytes=10_000).fetch("https://example.com/a.txt")


def test_fetch_reencodes_declared_charset_to_utf8():
    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/plain; charset=ISO-8859-1"},
            content="São Paulo".encode("latin-1"),
        )

    doc = make_fetcher(handler).fetch("https://example.com/sp")
    assert doc.data.decode("utf-8") == "São Paulo"


def test_fetch_ignores_unknown_charset():
    def handler(request):
        return httpx.Response(
            200, headers={"content-type": "text/plain; charset=bogus"}, content=b"hello"
        )

    assert make_fetcher(handler).fetch("https://example.com/x").data == b"hello"


def test_fetcher_validates_arguments():
    with pytest.raises(ValueError):
        Fetcher(max_bytes=0)
    with pytest.raises(ValueError):
        Fetcher(timeout=0)
    with pytest.raises(ValueError):
        Fetcher(max_redirects=-1)


# RAG.ingest_url


def _pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_ingest_url_indexes_html_with_final_url_as_source(rag):
    def handler(request):
        if request.url.path == "/lisbon":
            return httpx.Response(302, headers={"location": "https://example.com/cities/lisbon"})
        return html_page(request)

    rag.fetcher = make_fetcher(handler)
    info = rag.ingest_url("https://example.com/lisbon")

    assert info.source == "https://example.com/cities/lisbon"
    answer = rag.ask("Which city hosts Web Summit?")
    assert answer.citations[0].source == "https://example.com/cities/lisbon"
    assert all("tracking" not in hit.chunk.text for hit in rag.retrieve("tracking"))


def test_ingest_url_indexes_plain_text(rag):
    rag.fetcher = make_fetcher(
        lambda r: httpx.Response(200, headers={"content-type": "text/plain"}, text=LISBON)
    )
    info = rag.ingest_url("https://example.com/lisbon.txt")
    assert info.chunks >= 1
    assert rag.retrieve("Tagus river")[0].chunk.source == "https://example.com/lisbon.txt"


def test_ingest_url_reads_pdf(rag, monkeypatch):
    from citerag import loaders

    calls = []
    real_load = loaders.load

    def spy(filename, data):
        calls.append(filename)
        if filename.endswith(".pdf"):
            return [(1, LISBON)]
        return real_load(filename, data)

    monkeypatch.setattr(loaders, "load", spy)
    rag.fetcher = make_fetcher(
        lambda r: httpx.Response(
            200, headers={"content-type": "application/pdf"}, content=_pdf_bytes()
        )
    )
    info = rag.ingest_url("https://example.com/download?id=7")

    assert calls == ["download.pdf"]
    assert info.source == "https://example.com/download?id=7"
    assert rag.retrieve("Web Summit")[0].chunk.page == 1


def test_ingest_url_blocked_address_indexes_nothing(rag):
    rag.fetcher = make_fetcher(
        lambda r: pytest.fail("no request expected"), FakeDNS({"localhost": ["127.0.0.1"]})
    )
    with pytest.raises(BlockedURLError):
        rag.ingest_url("http://localhost:6333/collections")
    assert rag.documents() == []


def test_ingest_url_unsupported_content_type(rag):
    rag.fetcher = make_fetcher(
        lambda r: httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG")
    )
    with pytest.raises(UnsupportedFileError):
        rag.ingest_url("https://example.com/logo")


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(302, text="moved, but no Location"),
        httpx.Response(300, headers={"content-type": "text/html"}, text="<p>pick one</p>"),
        httpx.Response(304),
        httpx.Response(503),
    ],
)
def test_fetch_rejects_non_2xx_responses(response):
    with pytest.raises(URLFetchError, match=f"HTTP {response.status_code}"):
        make_fetcher(lambda r: response).fetch("https://example.com/page")


def test_fetch_rejects_non_string_url():
    with pytest.raises(BlockedURLError):
        make_fetcher(html_page).fetch(None)


def test_fetch_wraps_timeouts():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(URLFetchError, match="Timed out"):
        make_fetcher(handler).fetch("https://example.com/")


def test_fetch_passes_remaining_time_to_each_request():
    timeouts = []

    def handler(request):
        timeouts.append(request.extensions["timeout"]["read"])
        return html_page(request)

    make_fetcher(handler, timeout=5).fetch("https://example.com/")
    assert 0 < timeouts[0] <= 5


def test_fetch_deadline_covers_slow_body_and_redirect_chains(monkeypatch):
    from citerag import fetch

    clock = [0.0]
    monkeypatch.setattr(fetch.time, "monotonic", lambda: clock[0])

    class Drip(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(100):
                clock[0] += 1
                yield b"x"

    def slow_body(request):
        return httpx.Response(200, headers={"content-type": "text/plain"}, stream=Drip())

    with pytest.raises(URLFetchError, match="Timed out"):
        make_fetcher(slow_body, timeout=10).fetch("https://example.com/a.txt")

    def slow_redirect(request):
        clock[0] += 4
        return httpx.Response(302, headers={"location": "/next"})

    clock[0] = 0.0
    with pytest.raises(URLFetchError, match="Timed out"):
        make_fetcher(slow_redirect, timeout=10).fetch("https://example.com/")
    assert clock[0] == 12  # three hops, then the deadline stops the fourth


def test_fetch_uses_html_meta_charset_without_header_charset():
    page = '<html><head><meta charset="windows-1252"></head><p>Café</p></html>'

    def handler(request):
        return httpx.Response(
            200, headers={"content-type": "text/html"}, content=page.encode("cp1252")
        )

    assert "Café" in make_fetcher(handler).fetch("https://example.com/").data.decode("utf-8")
