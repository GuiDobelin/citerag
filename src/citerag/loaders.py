"""Turn uploaded files into (page, text) pairs."""

import io
import zipfile
import zlib
from html.parser import HTMLParser
from pathlib import PurePath
from xml.etree import ElementTree

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".html", ".htm", ".docx"}

# Upper bound for the uncompressed body of a DOCX file, so a small ZIP can't expand into gigabytes
# of XML (a "ZIP bomb"). Documents of several hundred pages stay well below this.
MAX_DOCX_XML_BYTES = 50 * 1024 * 1024

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class UnsupportedFileError(ValueError):
    pass


class InvalidFileError(ValueError):
    """The file has a supported type but its content can't be read."""


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


def _docx_xml(data: bytes) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            try:
                info = archive.getinfo("word/document.xml")
            except KeyError:
                raise InvalidFileError(
                    "Not a Word document: 'word/document.xml' is missing."
                ) from None
            # The size in the ZIP header can lie, so also cap what is actually decompressed.
            if info.file_size > MAX_DOCX_XML_BYTES:
                raise InvalidFileError("Word document body is too large to index.")
            with archive.open(info) as body:
                xml = body.read(MAX_DOCX_XML_BYTES + 1)
    except (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError, RuntimeError) as exc:
        raise InvalidFileError(f"Corrupt or unreadable DOCX file: {exc}") from exc
    if len(xml) > MAX_DOCX_XML_BYTES:
        raise InvalidFileError("Word document body is too large to index.")
    return xml


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    parts = []
    # iter() also walks runs nested in hyperlinks, tracked insertions and fields. Deleted text is
    # stored as w:delText and field codes as w:instrText, so neither is picked up.
    for el in paragraph.iter():
        if el.tag == f"{_W}t":
            parts.append(el.text or "")
        elif el.tag == f"{_W}tab":
            parts.append("\t")
        elif el.tag in (f"{_W}br", f"{_W}cr"):
            parts.append("\n")
        elif el.tag == f"{_W}noBreakHyphen":
            parts.append("-")
    return "".join(parts).strip()


def _block_lines(container: ElementTree.Element) -> list[str]:
    """Text of the paragraphs and tables in `container`, in document order, one line per block."""
    lines = []
    for el in container:
        if el.tag == f"{_W}p":
            text = _paragraph_text(el)
            if text:
                lines.append(text)
        elif el.tag == f"{_W}tbl":
            for row in el.findall(f"{_W}tr"):
                cells = [" ".join(_block_lines(cell)) for cell in row.findall(f"{_W}tc")]
                if any(cells):
                    lines.append(" | ".join(cells))
        else:
            # Content controls (w:sdt/w:sdtContent), custom XML and similar wrappers.
            lines.extend(_block_lines(el))
    return lines


def _docx_text(data: bytes) -> str:
    xml = _docx_xml(data)
    # Word never writes a DTD; refusing one rules out entity-expansion tricks in the XML parser.
    if b"<!DOCTYPE" in xml:
        raise InvalidFileError("Word document body contains a DTD, which is not allowed.")
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise InvalidFileError(f"Corrupt DOCX file: {exc}") from exc
    body = root.find(f"{_W}body")
    return "\n".join(_block_lines(body)) if body is not None else ""


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

    if suffix == ".docx":
        text = _docx_text(data)
        return [(None, text)] if text.strip() else []

    text = data.decode("utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        parser = _TextExtractor()
        parser.feed(text)
        text = "\n".join(parser.parts)
    return [(None, text)] if text.strip() else []
