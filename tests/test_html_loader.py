import pytest

from citerag.loaders import load


@pytest.mark.parametrize(
    "tag",
    [
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
    ],
)
def test_skips_boilerplate_including_nested_elements(tag):
    html = f"<{tag}><div>unwanted</div></{tag}><p>Useful text</p>"
    assert load("page.html", html.encode()) == [(None, "Useful text")]


@pytest.mark.parametrize("container", ["main", "article"])
def test_prefers_content_region(container):
    html = f"<p>Outside</p><{container}><p>Useful <b>text</b></p></{container}><p>After</p>"
    assert load("page.html", html.encode()) == [(None, "Useful text")]


def test_main_includes_nested_articles_once_and_excludes_articles_outside():
    html = b"<article>Outside</article><main><article>Inside</article><p>End</p></main>"
    assert load("page.html", html) == [(None, "Inside\nEnd")]


def test_multiple_articles_and_nested_articles_keep_order_without_duplicates():
    html = b"<p>Outside</p><article>One<article>Two</article></article><article>Three</article>"
    assert load("page.html", html) == [(None, "One\nTwo\nThree")]


def test_preserves_blocks_and_keeps_inline_text_together():
    html = (
        b"<h1>Title</h1><p>Hello <b>world</b> &amp; <em>friends</em></p>"
        b"<div>Next<br>line</div><ul><li>One</li><li>Two</li></ul>"
        b"<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>"
    )
    assert load("page.html", html) == [
        (None, "Title\nHello world & friends\nNext\nline\nOne\nTwo\nA B\nC")
    ]


@pytest.mark.parametrize(
    "meta",
    [
        '<meta charset="windows-1252">',
        "<META CHARSET=windows-1252/>",
        '<meta http-equiv="Content-Type" content="text/html; charset=windows-1252">',
    ],
)
def test_respects_meta_charset(meta):
    html = f"<html><head>{meta}</head><body><p>café €</p></body></html>"
    assert load("page.htm", html.encode("windows-1252")) == [(None, "café €")]


@pytest.mark.parametrize("charset", ["unknown-encoding", "base64", "rot_13"])
def test_invalid_or_nontext_charset_falls_back_to_utf8(charset):
    html = f'<meta charset="{charset}"><p>café</p>'.encode()
    assert load("page.html", html) == [(None, "café")]


def test_empty_main_does_not_fall_back_to_boilerplate():
    assert load("page.html", b"<p>Outside</p><main><nav>Menu</nav></main>") == []


def test_self_closing_skipped_element_does_not_hide_following_text():
    assert load("page.html", b"<svg/><p>Useful</p>") == [(None, "Useful")]


def test_unclosed_inline_element_does_not_hide_later_content():
    assert load("page.html", b"<main><p>Hello <b>world</p><p>Next</main>After") == [
        (None, "Hello world\nNext")
    ]


def test_malformed_meta_attributes_fall_back_to_utf8():
    assert load("page.html", b"<meta http-equiv content><p>Useful</p>") == [(None, "Useful")]


def test_regions_in_skipped_elements_do_not_select_content():
    html = b"<nav><main>Menu</main></nav><p>Useful</p>"
    assert load("page.html", html) == [(None, "Useful")]


def test_html_ingestion_indexes_content_without_boilerplate(rag):
    html = b"<nav>Menu</nav><main><p>Lisbon is the <b>capital</b> of Portugal.</p></main>"
    document = rag.ingest("page.html", html)
    chunks = rag.store.chunks()
    assert document.chunks == 1
    assert len(chunks) == 1
    assert chunks[0].source == "page.html"
    assert chunks[0].text == "Lisbon is the capital of Portugal."


def test_source_whitespace_does_not_split_a_paragraph():
    html = b"<p>Hello\n    <b>wide\nworld</b></p><p>Next paragraph</p>"
    assert load("page.html", html) == [(None, "Hello wide world\nNext paragraph")]


def test_utf8_bom_wins_over_meta_charset():
    html = '\ufeff<meta charset="windows-1252"><p>café</p>'.encode()
    assert load("page.html", html) == [(None, "café")]
