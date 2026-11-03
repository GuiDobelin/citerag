from typer.testing import CliRunner

from citerag.cli import app
from citerag.pipeline import RAG

runner = CliRunner()


def test_list_documents(rag, monkeypatch):
    rag.ingest("lisbon.txt", b"Lisbon is the capital of Portugal.")
    rag.ingest("berlin.txt", b"Berlin is the capital of Germany.")
    monkeypatch.setattr(RAG, "from_settings", lambda: rag)

    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    documents = rag.store.documents()
    lines = result.output.splitlines()
    assert len(lines) == len(documents) == 2
    for document in documents:
        assert (
            f"{document.doc_id}  {document.source}  ({document.chunks} chunks)" in lines
        )


def test_list_documents_when_store_is_empty(rag, monkeypatch):
    monkeypatch.setattr(RAG, "from_settings", lambda: rag)

    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert result.output == ""


def test_delete_document(rag, monkeypatch):
    document = rag.ingest("lisbon.txt", b"Lisbon is the capital of Portugal.")
    monkeypatch.setattr(RAG, "from_settings", lambda: rag)

    result = runner.invoke(app, ["delete", document.doc_id])

    assert result.exit_code == 0
    assert f"Deleted document: {document.doc_id}" in result.output
    assert rag.store.documents() == []


def test_delete_missing_document_returns_nonzero(rag, monkeypatch):
    monkeypatch.setattr(RAG, "from_settings", lambda: rag)

    result = runner.invoke(app, ["delete", "missing-id"])

    assert result.exit_code == 1
    assert "Document not found: missing-id" in result.output


def test_cli_help_includes_list_and_delete():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "list" in result.output
    assert "delete" in result.output
