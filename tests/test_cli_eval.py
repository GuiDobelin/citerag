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
    assert "nDCG@2" in result.stdout
    assert "Hit rate@2" in result.stdout
    assert "2 questions" in result.stdout
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
    # One mode still prints an array, so scripts parse every run the same way.
    [payload] = json.loads(result.stdout)
    assert payload["mode"] == "keyword"
    assert payload["recall"] == 1.0
    assert payload["mrr"] == 1.0
    assert payload["ndcg"] == 1.0
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


def test_eval_defaults_to_the_configured_top_k_and_mode(use_rag, tmp_path):
    use_rag.retrieval_mode = "keyword"
    dataset = _write_dataset(tmp_path, [{"question": "Web Summit?", "relevant": ["lisbon.txt"]}])
    result = runner.invoke(app, ["eval", str(dataset), "--json"])
    assert result.exit_code == 0, result.output
    [payload] = json.loads(result.stdout)
    assert payload["k"] == use_rag.top_k == 3
    assert payload["mode"] == "keyword"
    assert "Recall@3" in runner.invoke(app, ["eval", str(dataset)]).stdout


def test_eval_does_not_warn_when_every_source_is_indexed(use_rag, tmp_path):
    dataset = _write_dataset(
        tmp_path,
        [{"question": "Web Summit?", "relevant": ["lisbon.txt", {"source": "berlin.md"}]}],
    )
    result = runner.invoke(app, ["eval", str(dataset)])
    assert result.exit_code == 0, result.output
    assert "warning" not in result.output


def test_eval_notes_when_a_reranker_is_configured(use_rag, tmp_path):
    class ReverseReranker:
        def score(self, question, texts):
            return [-float(i) for i in range(len(texts))]

    dataset = _write_dataset(tmp_path, [{"question": "Web Summit?", "relevant": ["lisbon.txt"]}])
    assert "reranked" not in runner.invoke(app, ["eval", str(dataset)]).stdout
    use_rag.reranker = ReverseReranker()
    assert "reranked" in runner.invoke(app, ["eval", str(dataset)]).stdout


def test_eval_exits_one_when_the_dataset_is_missing(use_rag, tmp_path):
    result = runner.invoke(app, ["eval", str(tmp_path / "nope.jsonl")])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)  # a clean exit, not an uncaught error
    assert "nope.jsonl" in result.output


def test_eval_and_ask_reject_unknown_modes_the_same_way(use_rag, tmp_path):
    dataset = _write_dataset(tmp_path, [{"question": "Q", "relevant": ["lisbon.txt"]}])
    evaluated = runner.invoke(app, ["eval", str(dataset), "--mode", "fuzzy"])
    asked = runner.invoke(app, ["ask", "Q", "--mode", "fuzzy"])
    assert evaluated.exit_code == asked.exit_code == 2
    assert "--mode" in evaluated.output and "--mode" in asked.output
    assert "vector, keyword, hybrid, all" in evaluated.output
    assert "vector, keyword, hybrid" in asked.output
    assert "hybrid, all" not in asked.output


def test_eval_pluralises_the_question_count(use_rag, tmp_path):
    dataset = _write_dataset(tmp_path, [{"question": "Web Summit?", "relevant": ["lisbon.txt"]}])
    assert "\n1 question\n" in runner.invoke(app, ["eval", str(dataset)]).stdout
