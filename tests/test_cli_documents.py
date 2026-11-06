from conftest import BERLIN, LISBON
from typer.testing import CliRunner

from citerag.cli import app
from citerag.embeddings import HashEmbedder
from citerag.llm import ExtractiveGenerator
from citerag.pipeline import RAG
from citerag.store import MemoryStore

runner = CliRunner()


def _use(rag, monkeypatch) -> RAG:
    monkeypatch.setattr(RAG, "from_settings", lambda: rag)
    return rag


def test_help_lists_the_document_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "list" in result.stdout
    assert "delete" in result.stdout


def test_list_with_no_documents(rag, monkeypatch):
    _use(rag, monkeypatch)
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert result.stdout == "No documents indexed.\n"


def test_list_prints_one_line_per_document(rag, monkeypatch):
    lisbon = rag.ingest("lisbon.txt", LISBON.encode())
    berlin = rag.ingest("berlin.md", BERLIN.encode())
    _use(rag, monkeypatch)
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert result.stdout.splitlines() == [
        f"{lisbon.doc_id}  lisbon.txt  ({lisbon.chunks} chunks)",
        f"{berlin.doc_id}  berlin.md  ({berlin.chunks} chunks)",
    ]


def test_list_keeps_each_document_on_one_line_and_strips_control_characters(rag, monkeypatch):
    info = rag.ingest("evil\n\x1b[31mname.txt", LISBON.encode())
    _use(rag, monkeypatch)
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert result.stdout == f"{info.doc_id}  evil [31mname.txt  ({info.chunks} chunks)\n"


def test_delete_removes_the_document_from_the_store_and_keyword_index(rag, monkeypatch):
    lisbon = rag.ingest("lisbon.txt", LISBON.encode())
    berlin = rag.ingest("berlin.md", BERLIN.encode())
    rag.retrieval_mode = "keyword"
    assert any(h.chunk.doc_id == lisbon.doc_id for h in rag.retrieve("Web Summit Lisbon"))
    _use(rag, monkeypatch)

    result = runner.invoke(app, ["delete", lisbon.doc_id])

    assert result.exit_code == 0
    assert result.stdout == f"deleted  {lisbon.doc_id}\n"
    assert [d.doc_id for d in rag.documents()] == [berlin.doc_id]
    assert all(h.chunk.doc_id != lisbon.doc_id for h in rag.retrieve("Web Summit Lisbon"))
    listed = runner.invoke(app, ["list"])
    assert lisbon.doc_id not in listed.stdout
    assert berlin.doc_id in listed.stdout


def test_delete_is_persisted_across_cli_runs(tmp_path, monkeypatch):
    def fresh_rag() -> RAG:
        # Each CLI command builds its own RAG from disk, as `RAG.from_settings` does.
        return RAG(HashEmbedder(), MemoryStore(tmp_path / "store"), ExtractiveGenerator())

    info = fresh_rag().ingest("lisbon.txt", LISBON.encode())
    monkeypatch.setattr(RAG, "from_settings", fresh_rag)

    assert runner.invoke(app, ["delete", info.doc_id]).exit_code == 0
    listed = runner.invoke(app, ["list"])
    assert listed.exit_code == 0
    assert listed.stdout == "No documents indexed.\n"
    assert runner.invoke(app, ["delete", info.doc_id]).exit_code == 1


def test_delete_unknown_id_fails(rag, monkeypatch):
    info = rag.ingest("lisbon.txt", LISBON.encode())
    _use(rag, monkeypatch)
    result = runner.invoke(app, ["delete", "nope"])
    assert result.exit_code == 1
    assert "No document with id 'nope'." in result.stderr
    assert result.stdout == ""
    assert [d.doc_id for d in rag.documents()] == [info.doc_id]
