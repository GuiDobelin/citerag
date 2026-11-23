import json

import pytest
from conftest import BERLIN, LISBON
from typer.testing import CliRunner

from citerag.cli import app
from citerag.pipeline import RAG

runner = CliRunner()


@pytest.fixture
def use_rag(rag, monkeypatch):
    rag.ingest("lisbon.txt", LISBON.encode())
    rag.ingest("berlin.md", BERLIN.encode())
    monkeypatch.setattr(RAG, "from_settings", lambda: rag)
    return rag


def _write_dataset(tmp_path, rows):
    path = tmp_path / "qa.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_eval_prints_metrics_table_and_details(use_rag, tmp_path):
    dataset = _write_dataset(
        tmp_path,
        [
            {
                "id": "summit",
                "question": "Which city hosts Web Summit?",
                "relevant": [{"source": "lisbon.txt", "contains": "Web Summit"}],
            },
            {
                "id": "missing",
                "question": "Which city hosts Lisbon?",
                "relevant": ["not_indexed.txt"],
            },
        ],
    )
    result = runner.invoke(
        app, ["eval", str(dataset), "--k", "2", "--mode", "keyword", "--details"]
    )
    assert result.exit_code == 0, result.output
    assert "warning: source 'not_indexed.txt' is not indexed." in result.output
    assert "Recall@2" in result.stdout
    assert "MRR@2" in result.stdout
    assert "Hit rate@2" in result.stdout
    assert "keyword" in result.stdout
    assert "Missed (keyword):" in result.stdout
    assert "missing: retrieved [" in result.stdout


def test_eval_mode_all_and_json_output(use_rag, tmp_path):
    dataset = _write_dataset(
        tmp_path,
        [{"question": "Which city hosts Web Summit?", "relevant": ["lisbon.txt"]}],
    )
    table = runner.invoke(app, ["eval", str(dataset), "--k", "3", "--mode", "all"])
    assert table.exit_code == 0, table.output
    for mode in ("vector", "keyword", "hybrid"):
        assert mode in table.stdout

    as_json = runner.invoke(app, ["eval", str(dataset), "--k", "3", "--mode", "all", "--json"])
    assert as_json.exit_code == 0, as_json.output
    payload = json.loads(as_json.stdout)
    assert [entry["mode"] for entry in payload] == ["vector", "keyword", "hybrid"]
    assert all(entry["k"] == 3 and entry["questions"] == 1 for entry in payload)


def test_eval_single_mode_json_output(use_rag, tmp_path):
    dataset = _write_dataset(
        tmp_path,
        [{"question": "Where do companies hire Python developers?", "relevant": ["berlin.md"]}],
    )
    result = runner.invoke(app, ["eval", str(dataset), "--mode", "keyword", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["mode"] == "keyword"
    assert payload["recall"] == 1.0
    assert payload["mrr"] == 1.0
    assert payload["hit_rate"] == 1.0


def test_eval_exits_one_on_invalid_dataset_with_line_number(use_rag, tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"question": "Q", "relevant": ["lisbon.txt"]}\n{bad json\n', encoding="utf-8")
    result = runner.invoke(app, ["eval", str(bad)])
    assert result.exit_code == 1
    assert "line 2" in result.output


def test_eval_rejects_unknown_mode(use_rag, tmp_path):
    dataset = _write_dataset(tmp_path, [{"question": "Q", "relevant": ["lisbon.txt"]}])
    result = runner.invoke(app, ["eval", str(dataset), "--mode", "fuzzy"])
    assert result.exit_code == 2
