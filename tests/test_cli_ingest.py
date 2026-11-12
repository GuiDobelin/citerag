import httpx
import pytest
from conftest import BERLIN, LISBON, FakeDNS, make_fetcher
from typer.testing import CliRunner

from citerag.cli import app
from citerag.pipeline import RAG

runner = CliRunner()


def html(text: str) -> httpx.Response:
    return httpx.Response(
        200, headers={"content-type": "text/html; charset=utf-8"}, text=f"<p>{text}</p>"
    )


@pytest.fixture
def cli_rag(rag, monkeypatch) -> RAG:
    monkeypatch.setattr(RAG, "from_settings", lambda: rag)
    rag.fetcher = make_fetcher(lambda r: pytest.fail(f"unexpected request to {r.url}"))
    return rag


def test_ingest_files_and_folders(cli_rag, tmp_path):
    docs = tmp_path / "docs"
    (docs / "sub").mkdir(parents=True)
    (docs / "sub" / "lisbon.txt").write_text(LISBON)
    single = tmp_path / "berlin.md"
    single.write_text(BERLIN)

    result = runner.invoke(app, ["ingest", str(docs), str(single)])

    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert [line.split("  ")[1] for line in lines] == [
        str(docs / "sub" / "lisbon.txt"),
        str(single),
    ]
    assert sorted(d.source for d in cli_rag.documents()) == ["berlin.md", "lisbon.txt"]


def test_ingest_mixes_urls_with_files_in_argument_order(cli_rag, tmp_path):
    cli_rag.fetcher = make_fetcher(lambda r: html(LISBON))
    berlin = tmp_path / "berlin.md"
    berlin.write_text(BERLIN)

    result = runner.invoke(app, ["ingest", "https://example.com/lisbon", str(berlin)])

    assert result.exit_code == 0, result.output
    first, second = result.stdout.splitlines()
    assert first.startswith("indexed  https://example.com/lisbon  (")
    assert second.startswith(f"indexed  {berlin}  (")
    sources = {d.source for d in cli_rag.documents()}
    assert sources == {"https://example.com/lisbon", "berlin.md"}


def test_ingest_url_shows_the_final_address_after_a_redirect(cli_rag):
    def handler(request):
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/new"})
        return html(LISBON)

    cli_rag.fetcher = make_fetcher(handler)
    result = runner.invoke(app, ["ingest", "HTTPS://example.com/old"])

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith(
        "indexed  HTTPS://example.com/old -> https://example.com/new  ("
    )


def test_ingest_skips_a_blocked_url_and_continues(cli_rag, tmp_path):
    cli_rag.fetcher = make_fetcher(
        lambda r: pytest.fail("no request expected"), FakeDNS({"intranet": ["10.0.0.5"]})
    )
    berlin = tmp_path / "berlin.md"
    berlin.write_text(BERLIN)

    result = runner.invoke(app, ["ingest", "http://intranet/wiki", str(berlin)])

    assert result.exit_code == 0, result.output
    skipped, indexed = result.stdout.splitlines()
    assert skipped.startswith("skipped  http://intranet/wiki: ")
    assert "CITERAG_ALLOW_PRIVATE_URLS" in skipped
    assert indexed.startswith(f"indexed  {berlin}")
    assert [d.source for d in cli_rag.documents()] == ["berlin.md"]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(404),
        httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG"),
    ],
)
def test_ingest_reports_url_errors_as_skipped(cli_rag, response):
    cli_rag.fetcher = make_fetcher(lambda r: response)
    result = runner.invoke(app, ["ingest", "https://example.com/x"])
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("skipped  https://example.com/x: ")
    assert cli_rag.documents() == []


def test_ingest_strips_control_characters_from_messages(cli_rag):
    cli_rag.fetcher = make_fetcher(lambda r: httpx.Response(503))
    result = runner.invoke(app, ["ingest", "https://example.com/\x1b[31mred"])
    assert "\x1b" not in result.stdout
    assert len(result.stdout.splitlines()) == 1


def test_ingest_checks_every_path_before_indexing_anything(cli_rag, tmp_path):
    berlin = tmp_path / "berlin.md"
    berlin.write_text(BERLIN)

    result = runner.invoke(app, ["ingest", str(berlin), str(tmp_path / "missing.txt")])

    assert result.exit_code == 2
    assert "does not exist" in result.output
    assert cli_rag.documents() == []


def test_ingest_requires_at_least_one_source():
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 2
