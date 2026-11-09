import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from citerag import loaders
from citerag.api import create_app, get_rag

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def para(*runs: str) -> str:
    """A w:p whose runs are given as raw run content, e.g. '<w:t>Hi</w:t>'."""
    return "<w:p>" + "".join(f"<w:r>{r}</w:r>" for r in runs) + "</w:p>"


def text(value: str) -> str:
    return f'<w:t xml:space="preserve">{value}</w:t>'


def table(*rows: list[str]) -> str:
    cells = (
        "<w:tr>" + "".join(f"<w:tc>{para(text(c))}</w:tc>" for c in row) + "</w:tr>" for row in rows
    )
    return "<w:tbl>" + "".join(cells) + "</w:tbl>"


def make_docx(body: str, *, prolog: str = "") -> bytes:
    xml = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>{prolog}'
        f'<w:document xmlns:w="{W_NS}"><w:body>{body}<w:sectPr/></w:body></w:document>'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", xml)
    return buf.getvalue()


def load_text(data: bytes) -> str:
    [(page, content)] = loaders.load("report.docx", data)
    assert page is None
    return content


def test_docx_paragraphs_in_order():
    data = make_docx(para(text("Termination")) + para(text("Either party may "), text("end it.")))
    assert load_text(data) == "Termination\nEither party may end it."


def test_docx_tabs_breaks_and_hyphens():
    data = make_docx(
        para(
            text("a"), "<w:tab/>", text("b"), "<w:br/>", text("c"), "<w:noBreakHyphen/>", text("d")
        )
    )
    assert load_text(data) == "a\tb\nc-d"


def test_docx_tables_row_by_row():
    data = make_docx(para(text("Prices")) + table(["Plan", "EUR"], ["Pro", "20"]))
    assert load_text(data) == "Prices\nPlan | EUR\nPro | 20"


def test_docx_nested_runs_and_content_controls():
    hyperlink = f"<w:p><w:hyperlink><w:r>{text('see docs')}</w:r></w:hyperlink></w:p>"
    control = f"<w:sdt><w:sdtContent>{para(text('inside control'))}</w:sdtContent></w:sdt>"
    assert load_text(make_docx(hyperlink + control)) == "see docs\ninside control"


def test_docx_skips_deleted_text_and_field_codes():
    tracked = (
        f"<w:p><w:del><w:r><w:delText>old</w:delText></w:r></w:del>"
        f"<w:ins><w:r>{text('new')}</w:r></w:ins>"
        f"<w:r><w:instrText>PAGE</w:instrText></w:r></w:p>"
    )
    assert load_text(make_docx(tracked)) == "new"


def test_docx_suffix_is_case_insensitive():
    assert loaders.load("REPORT.DOCX", make_docx(para(text("hi")))) == [(None, "hi")]


def test_docx_without_text_is_empty():
    assert loaders.load("empty.docx", make_docx(para() + "<w:p/>")) == []


@pytest.mark.parametrize(
    "data",
    [b"not a zip", b"PK\x03\x04garbage", make_docx(para(text("x")))[:-30]],
    ids=["not-zip", "bad-header", "truncated"],
)
def test_docx_corrupt_archive(data):
    with pytest.raises(loaders.InvalidFileError):
        loaders.load("broken.docx", data)


def test_docx_missing_document_xml():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("xl/workbook.xml", "<workbook/>")
    with pytest.raises(loaders.InvalidFileError, match="document.xml"):
        loaders.load("sheet.docx", buf.getvalue())


def test_docx_malformed_xml():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "<w:document")
    with pytest.raises(loaders.InvalidFileError):
        loaders.load("bad.docx", buf.getvalue())


def test_docx_rejects_dtd():
    bomb = '<!DOCTYPE w:document [<!ENTITY a "aaaaaaaaaa">]>'
    with pytest.raises(loaders.InvalidFileError, match="DTD"):
        loaders.load("bomb.docx", make_docx(para(text("&a;")), prolog=bomb))


def test_docx_size_limit(monkeypatch):
    monkeypatch.setattr(loaders, "MAX_DOCX_XML_BYTES", 1000)
    data = make_docx(para(text("x" * 2000)))
    with pytest.raises(loaders.InvalidFileError, match="too large"):
        loaders.load("big.docx", data)


def test_docx_size_limit_ignores_lying_header(monkeypatch):
    data = bytearray(make_docx(para(text("x" * 2000))))
    # Patch the uncompressed size in the central directory entry to claim a tiny file.
    with zipfile.ZipFile(io.BytesIO(bytes(data))) as zf:
        info = zf.getinfo("word/document.xml")
    cd = data.rfind(b"PK\x01\x02", 0, data.rfind(b"word/document.xml"))
    data[cd + 24 : cd + 28] = (10).to_bytes(4, "little")
    with zipfile.ZipFile(io.BytesIO(bytes(data))) as zf:
        assert zf.getinfo("word/document.xml").file_size == 10 < info.file_size
    monkeypatch.setattr(loaders, "MAX_DOCX_XML_BYTES", 1000)
    with pytest.raises(loaders.InvalidFileError):
        loaders.load("liar.docx", bytes(data))


def test_api_accepts_docx_and_rejects_corrupt(rag):
    app = create_app()
    app.dependency_overrides[get_rag] = lambda: rag
    client = TestClient(app)
    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    data = make_docx(para(text("Lisbon hosts Web Summit every November.")))
    r = client.post("/documents", files={"file": ("summit.docx", data, mime)})
    assert r.status_code == 201
    answer = client.post("/query", json={"question": "Where is Web Summit?"}).json()
    assert answer["citations"][0]["source"] == "summit.docx"

    r = client.post("/documents", files={"file": ("broken.docx", b"nope", mime)})
    assert r.status_code == 422
    assert "DOCX" in r.json()["detail"]
