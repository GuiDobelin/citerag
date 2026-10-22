"""Turn uploaded files into (page, text) pairs."""

import io
from html.parser import HTMLParser
from pathlib import PurePath

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".html", ".htm"}


class UnsupportedFileError(ValueError):
    pass


class _TextExtractor(HTMLParser):
    _skip = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._skip:
            self._depth += 1

    def handle_endtag(self, tag):
        if tag in self._skip and self._depth:
            self._depth -= 1

    def handle_data(self, data):
        if not self._depth and data.strip():
            self.parts.append(data.strip())


def load(filename: str, data: bytes) -> list[tuple[int | None, str]]:
    """Return the file's text as a list of (page_number, text). Page is None for non-paged files."""
    suffix = PurePath(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileError(
            f"Unsupported file type '{suffix or filename}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        pages = [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]
        return [(n, text) for n, text in pages if text.strip()]

    text = data.decode("utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        parser = _TextExtractor()
        parser.feed(text)
        text = "\n".join(parser.parts)
    return [(None, text)] if text.strip() else []
