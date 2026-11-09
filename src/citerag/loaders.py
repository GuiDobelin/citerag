"""Turn uploaded files into (page, text) pairs."""

import io
import re
import zipfile
import zlib
from html.parser import HTMLParser
from pathlib import PurePath
from xml.etree import ElementTree

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".html", ".htm", ".docx"}

# Limits for the body of a DOCX file, which is untrusted input. A small ZIP can expand into a huge
# XML file (a "ZIP bomb"), and a parsed element costs far more memory than its bytes, so both the
# size and the number of elements are capped. A 300-page report is around 5 MB and 150k elements.
MAX_DOCX_XML_BYTES = 20 * 1024 * 1024
MAX_DOCX_ELEMENTS = 1_000_000
# Real documents nest a handful of levels (tables in tables, content controls, text boxes).
MAX_DOCX_DEPTH = 64

# Transitional and Strict OOXML namespaces for WordprocessingML.
_W_NAMESPACES = (
    "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "http://purl.oclc.org/ooxml/wordprocessingml/main",
)
_MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
_DEFAULT_MAIN_PART = "word/document.xml"


class UnsupportedFileError(ValueError):
    pass


class InvalidFileError(ValueError):
    """The file has a supported type but its content can't be read."""


class _CharsetExtractor(HTMLParser):
    """Read HTML encoding declarations without decoding the document's text first."""

    def __init__(self) -> None:
        super().__init__()
        self.charset: str | None = None

    def handle_starttag(self, tag, attrs):
        if tag != "meta" or self.charset:
            return
        attrs = dict(attrs)
        self.charset = attrs.get("charset")
        if not self.charset and (attrs.get("http-equiv") or "").lower() == "content-type":
            match = re.search(r"charset\s*=\s*([^;\s]+)", attrs.get("content") or "", re.I)
            if match:
                self.charset = match[1].strip("\"'")


def _decode_html(data: bytes) -> str:
    parser = _CharsetExtractor()
    # Encoding declarations belong at the start of an HTML document. Latin-1 preserves all
    # ASCII markup while allowing us to inspect a bounded prefix of arbitrary input bytes.
    parser.feed(data[:4096].decode("latin-1"))
    try:
        return data.decode(parser.charset or "utf-8-sig", errors="replace")
    except (LookupError, UnicodeError):
        # Unknown labels and non-text codecs must not prevent an otherwise readable upload.
        return data.decode("utf-8-sig", errors="replace")


class _TextExtractor(HTMLParser):
    _skip = {
        "script",
        "style",
        "noscript",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "template",
        "svg",
        "iframe",
    }
    _blocks = {
        "p",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "div",
        "tr",
        "br",
        "main",
        "article",
        "section",
        "blockquote",
        "pre",
        "ul",
        "ol",
        "dl",
        "dt",
        "dd",
    }

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._main: list[str] = []
        self._article: list[str] = []
        self._stack: list[tuple[str, int, int, int]] = []
        self._skip_depth = self._main_depth = self._article_depth = 0
        self._has_main = self._has_article = False

    def _append(self, text: str) -> None:
        if self._skip_depth:
            return
        self.parts.append(text)
        if self._main_depth:
            self._main.append(text)
        if self._article_depth:
            self._article.append(text)

    def handle_starttag(self, tag, attrs):
        if tag in self._skip or tag in {"main", "article"}:
            self._stack.append((tag, self._skip_depth, self._main_depth, self._article_depth))
            if tag in self._skip:
                self._skip_depth += 1
            elif not self._skip_depth:
                if tag == "main":
                    self._main_depth += 1
                    self._has_main = True
                else:
                    self._article_depth += 1
                    self._has_article = True
        if tag in self._blocks:
            self._append("\n")

    def handle_endtag(self, tag):
        if tag in self._blocks:
            self._append("\n")
        # Track only content regions and skipped elements. Closing an enclosing region also
        # closes unclosed tracked children, so malformed inline markup cannot leak selection.
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                _, self._skip_depth, self._main_depth, self._article_depth = self._stack[i]
                del self._stack[i:]
                break

    def handle_data(self, data):
        if data.strip():
            self._append(" ".join(data.split()) + " ")

    def text(self) -> str:
        parts = self._main if self._has_main else self._article if self._has_article else self.parts
        return "\n".join(
            line for raw in "".join(parts).splitlines() if (line := " ".join(raw.split()))
        )


def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    info = archive.getinfo(name)
    if info.file_size > MAX_DOCX_XML_BYTES:
        raise InvalidFileError("Word document body is too large to index.")
    # zipfile stops at the declared size and checks the CRC, so a header that understates the size
    # fails as corrupt. The bounded read keeps the cap independent of that behaviour.
    with archive.open(info) as member:
        data = member.read(MAX_DOCX_XML_BYTES + 1)
    if len(data) > MAX_DOCX_XML_BYTES:
        raise InvalidFileError("Word document body is too large to index.")
    return data


def _main_part_name(archive: zipfile.ZipFile) -> str:
    """The main document part named in _rels/.rels, falling back to the usual word/document.xml."""
    try:
        rels = _parse_xml(_read_member(archive, "_rels/.rels"))
    except (KeyError, InvalidFileError):
        return _DEFAULT_MAIN_PART
    for rel in rels:
        if rel.get("Type", "").endswith("/officeDocument") and rel.get("Target"):
            return rel.get("Target").lstrip("/")
    return _DEFAULT_MAIN_PART


def _docx_xml(data: bytes) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            name = _main_part_name(archive)
            try:
                return _read_member(archive, name)
            except KeyError:
                raise InvalidFileError(f"Not a Word document: '{name}' is missing.") from None
    except (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError, RuntimeError) as exc:
        raise InvalidFileError(f"Corrupt or unreadable DOCX file: {exc}") from exc


def _parse_xml(xml: bytes) -> ElementTree.Element:
    # Word never writes a DTD; refusing one rules out entity-expansion tricks in the XML parser.
    # expat also accepts UTF-16, so look for the marker in those encodings too.
    if any("<!DOCTYPE".encode(enc) in xml for enc in ("utf-8", "utf-16-le", "utf-16-be")):
        raise InvalidFileError("Word document XML contains a DTD, which is not allowed.")
    parser = ElementTree.XMLPullParser(events=("start",))
    root = None
    count = 0
    step = 1024 * 1024
    try:
        for offset in range(0, len(xml), step):
            parser.feed(xml[offset : offset + step])
            for _, el in parser.read_events():
                root = el if root is None else root
                count += 1
            if count > MAX_DOCX_ELEMENTS:
                raise InvalidFileError("Word document has too many XML elements to index.")
        parser.close()
    except ElementTree.ParseError as exc:
        raise InvalidFileError(f"Corrupt DOCX file: {exc}") from exc
    if root is None:
        raise InvalidFileError("Corrupt DOCX file: empty XML.")
    return root


class _DocxText:
    """Collects the text of a WordprocessingML body, one line per paragraph or table row."""

    def __init__(self, namespace: str) -> None:
        w = f"{{{namespace}}}"
        self.p, self.tbl, self.tr, self.tc = f"{w}p", f"{w}tbl", f"{w}tr", f"{w}tc"
        self.t, self.tab, self.txbx = f"{w}t", f"{w}tab", f"{w}txbxContent"
        self.breaks = {f"{w}br", f"{w}cr"}
        self.hyphen = f"{w}noBreakHyphen"

    @staticmethod
    def _check_depth(depth: int) -> None:
        if depth > MAX_DOCX_DEPTH:
            raise InvalidFileError("Word document is nested too deeply to index.")

    def blocks(self, container: ElementTree.Element, depth: int = 0) -> list[str]:
        """Paragraphs and table rows in `container`, in document order."""
        self._check_depth(depth)
        lines = []
        for el in container:
            if el.tag == self.p:
                lines.extend(self._paragraph(el, depth + 1))
            elif el.tag == self.tbl:
                for row in self._find(el, self.tr, depth + 1):
                    cells = [
                        " ".join(self.blocks(cell, depth + 2))
                        for cell in self._find(row, self.tc, depth + 1)
                    ]
                    if any(cells):
                        lines.append(" | ".join(cells))
            elif el.tag != _MC_FALLBACK:
                # Content controls (w:sdt/w:sdtContent), custom XML and similar wrappers.
                lines.extend(self.blocks(el, depth + 1))
        return lines

    def _find(self, el: ElementTree.Element, tag: str, depth: int) -> list[ElementTree.Element]:
        """Children with `tag`, looking through wrappers such as content controls around rows."""
        self._check_depth(depth)
        found = []
        for child in el:
            if child.tag == tag:
                found.append(child)
            elif child.tag not in (self.tbl, _MC_FALLBACK):
                found.extend(self._find(child, tag, depth + 1))
        return found

    def _paragraph(self, paragraph: ElementTree.Element, depth: int) -> list[str]:
        parts: list[str] = []
        boxes: list[str] = []
        self._inline(paragraph, parts, boxes, depth)
        text = "".join(parts).strip()
        return ([text] if text else []) + boxes

    def _inline(self, el: ElementTree.Element, parts: list[str], boxes: list[str], depth: int):
        # Walks runs nested in hyperlinks, tracked insertions and fields. Deleted text is stored as
        # w:delText and field codes as w:instrText, so neither is picked up. Text boxes become lines
        # of their own, and only the preferred mc:Choice copy of alternate content is read.
        self._check_depth(depth)
        for child in el:
            tag = child.tag
            if tag == self.t:
                parts.append(child.text or "")
            elif tag == self.tab:
                parts.append("\t")
            elif tag in self.breaks:
                parts.append("\n")
            elif tag == self.hyphen:
                parts.append("-")
            elif tag == self.txbx:
                boxes.extend(self.blocks(child, depth + 1))
            elif tag != _MC_FALLBACK:
                self._inline(child, parts, boxes, depth + 1)


def _docx_text(data: bytes) -> str:
    root = _parse_xml(_docx_xml(data))
    namespace = root.tag[1:].partition("}")[0]
    if namespace not in _W_NAMESPACES:
        raise InvalidFileError("Not a Word document: unexpected main document XML.")
    body = root.find(f"{{{namespace}}}body")
    return "\n".join(_DocxText(namespace).blocks(body)) if body is not None else ""


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

    if suffix in {".html", ".htm"}:
        text = _decode_html(data)
        parser = _TextExtractor()
        parser.feed(text)
        parser.close()
        text = parser.text()
    else:
        text = data.decode("utf-8", errors="replace")
    return [(None, text)] if text.strip() else []
