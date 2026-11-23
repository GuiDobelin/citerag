import json

import pytest
from conftest import BERLIN, LISBON

from citerag.evaluation import (
    DatasetError,
    EvalCase,
    RelevantPassage,
    evaluate,
    load_dataset,
    parse_dataset,
    recall_at_k,
    reciprocal_rank,
)
from citerag.models import Chunk, Hit

LISBON_PASSAGE = RelevantPassage(source="lisbon.txt")
BERLIN_PASSAGE = RelevantPassage(source="berlin.md")


def hit(source: str, text: str = "some text", page: int | None = None, n: int = 0) -> Hit:
    chunk = Chunk(id=f"{source}:{n}", doc_id=source, source=source, page=page, text=text)
    return Hit(chunk=chunk, score=1.0)


def line(obj: object) -> str:
    return json.dumps(obj)


# --- matching ---------------------------------------------------------------


def test_source_only_passage_matches_any_chunk_of_that_source():
    assert LISBON_PASSAGE.matches(hit("lisbon.txt").chunk)
    assert not LISBON_PASSAGE.matches(hit("berlin.md").chunk)


def test_source_must_match_exactly():
    assert not RelevantPassage(source="lisbon").matches(hit("lisbon.txt").chunk)


def test_page_narrows_the_match():
    passage = RelevantPassage(source="report.pdf", page=3)
    assert passage.matches(hit("report.pdf", page=3).chunk)
    assert not passage.matches(hit("report.pdf", page=4).chunk)
    assert not passage.matches(hit("report.pdf", page=None).chunk)


def test_contains_ignores_case_and_whitespace():
    passage = RelevantPassage(source="a.txt", contains="web   SUMMIT\nconference")
    assert passage.matches(hit("a.txt", text="The Web Summit\tconference is big.").chunk)
    assert not passage.matches(hit("a.txt", text="The Web Summit is big.").chunk)


def test_contains_and_page_must_both_match():
    passage = RelevantPassage(source="r.pdf", page=2, contains="revenue")
    assert passage.matches(hit("r.pdf", text="Revenue grew", page=2).chunk)
    assert not passage.matches(hit("r.pdf", text="Revenue grew", page=1).chunk)
    assert not passage.matches(hit("r.pdf", text="Costs grew", page=2).chunk)


# --- metrics ----------------------------------------------------------------


def test_recall_counts_each_relevant_passage_once():
    hits = [hit("lisbon.txt", n=0), hit("lisbon.txt", n=1), hit("other.txt")]
    assert recall_at_k(hits, [LISBON_PASSAGE, BERLIN_PASSAGE], k=3) == 0.5


def test_recall_only_looks_at_the_top_k():
    hits = [hit("other.txt"), hit("lisbon.txt")]
    assert recall_at_k(hits, [LISBON_PASSAGE], k=1) == 0.0
    assert recall_at_k(hits, [LISBON_PASSAGE], k=2) == 1.0


def test_recall_with_k_larger_than_the_hit_list():
    assert recall_at_k([hit("lisbon.txt")], [LISBON_PASSAGE, BERLIN_PASSAGE], k=10) == 0.5
    assert recall_at_k([], [LISBON_PASSAGE], k=10) == 0.0


def test_recall_ignores_duplicate_relevant_passages():
    hits = [hit("lisbon.txt")]
    assert recall_at_k(hits, [LISBON_PASSAGE, LISBON_PASSAGE, BERLIN_PASSAGE], k=1) == 0.5


def test_one_hit_can_satisfy_several_passages():
    hits = [hit("a.txt", text="alpha and beta")]
    passages = [
        RelevantPassage(source="a.txt", contains="alpha"),
        RelevantPassage(source="a.txt", contains="beta"),
    ]
    assert recall_at_k(hits, passages, k=1) == 1.0


def test_reciprocal_rank_uses_the_first_relevant_hit():
    hits = [hit("x.txt"), hit("y.txt"), hit("lisbon.txt"), hit("lisbon.txt", n=1)]
    assert reciprocal_rank(hits, [LISBON_PASSAGE], k=5) == pytest.approx(1 / 3)
    assert reciprocal_rank(hits[2:], [LISBON_PASSAGE], k=5) == 1.0


def test_reciprocal_rank_is_zero_without_a_relevant_hit_in_the_top_k():
    hits = [hit("x.txt"), hit("lisbon.txt")]
    assert reciprocal_rank(hits, [LISBON_PASSAGE], k=1) == 0.0
    assert reciprocal_rank([], [LISBON_PASSAGE], k=1) == 0.0


@pytest.mark.parametrize("metric", [recall_at_k, reciprocal_rank])
def test_metrics_reject_bad_arguments(metric):
    with pytest.raises(ValueError, match="k must be"):
        metric([], [LISBON_PASSAGE], k=0)
    with pytest.raises(ValueError, match="relevant"):
        metric([], [], k=1)


# --- dataset ----------------------------------------------------------------


def test_parse_dataset_accepts_objects_and_shorthand_strings():
    cases = parse_dataset(
        [
            line({"question": "Q1", "relevant": [{"source": "a.pdf", "page": 2, "contains": "x"}]}),
            "",
            "   ",
            line({"question": "Q2", "relevant": ["b.md"], "id": "second"}),
        ]
    )
    assert cases == [
        EvalCase(question="Q1", relevant=[RelevantPassage(source="a.pdf", page=2, contains="x")]),
        EvalCase(question="Q2", relevant=[RelevantPassage(source="b.md")], id="second"),
    ]


def test_parse_dataset_deduplicates_relevant_passages():
    [case] = parse_dataset([line({"question": "Q", "relevant": ["a.md", {"source": "a.md"}]})])
    assert case.relevant == [RelevantPassage(source="a.md")]


@pytest.mark.parametrize(
    ("bad_line", "message"),
    [
        ("{not json", "invalid JSON"),
        (line(["a list"]), "JSON object"),
        (line({"relevant": ["a.md"]}), "question"),
        (line({"question": "  ", "relevant": ["a.md"]}), "question"),
        (line({"question": "Q"}), "relevant"),
        (line({"question": "Q", "relevant": []}), "relevant"),
        (line({"question": "Q", "relevant": [""]}), "source"),
        (line({"question": "Q", "relevant": [{"source": "a", "page": 0}]}), "page"),
        (line({"question": "Q", "relevant": [{"source": "a", "contains": " \n"}]}), "contains"),
        (line({"question": "Q", "relevant": [{"source": "a", "contain": "typo"}]}), "contain"),
        (line({"question": "Q", "relevant": ["a"], "answer": "unknown field"}), "answer"),
    ],
)
def test_parse_dataset_reports_the_bad_line(bad_line, message):
    good = line({"question": "Q", "relevant": ["a.md"]})
    with pytest.raises(DatasetError, match="line 3") as exc_info:
        parse_dataset([good, "", bad_line])
    assert message in str(exc_info.value)


def test_parse_dataset_rejects_an_empty_dataset():
    with pytest.raises(DatasetError, match="no questions"):
        parse_dataset(["", "  "])


def test_load_dataset_reads_utf8_with_bom(tmp_path):
    path = tmp_path / "qa.jsonl"
    content = line({"question": "Onde fica Lisboa?", "relevant": ["lisboa.txt"]}) + "\n"
    path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
    [case] = load_dataset(path)
    assert case.question == "Onde fica Lisboa?"


def test_load_dataset_rejects_non_utf8(tmp_path):
    path = tmp_path / "qa.jsonl"
    path.write_bytes(b'{"question": "caf\xe9", "relevant": ["a"]}')
    with pytest.raises(DatasetError, match="UTF-8"):
        load_dataset(path)


# --- evaluate ---------------------------------------------------------------

CASES = [
    EvalCase(
        id="summit",
        question="Which city hosts Web Summit?",
        relevant=[RelevantPassage(source="lisbon.txt", contains="Web Summit")],
    ),
    EvalCase(
        question="Where do companies hire Python developers?",
        relevant=[RelevantPassage(source="berlin.md")],
    ),
    EvalCase(question="Unanswerable", relevant=[RelevantPassage(source="missing.txt")]),
]


@pytest.fixture
def indexed(rag):
    rag.ingest("lisbon.txt", LISBON.encode())
    rag.ingest("berlin.md", BERLIN.encode())
    return rag


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_evaluate_scores_every_mode(indexed, mode):
    report = evaluate(indexed, CASES, k=3, mode=mode)
    assert report.mode == mode
    assert report.k == 3
    assert report.questions == 3
    assert not report.reranked
    summit, python, missing = report.results
    assert summit.id == "summit"
    assert summit.hit and summit.recall == 1.0
    assert missing.recall == 0.0 and missing.reciprocal_rank == 0.0 and not missing.hit
    assert all(not r.relevant for r in missing.retrieved)
    assert len(summit.retrieved) <= 3
    assert report.recall == pytest.approx(sum(r.recall for r in report.results) / 3)
    assert report.mrr == pytest.approx(sum(r.reciprocal_rank for r in report.results) / 3)
    assert report.hit_rate == pytest.approx(sum(r.hit for r in report.results) / 3)


def test_evaluate_keyword_mode_ranks_exact_terms_first(indexed):
    report = evaluate(indexed, CASES[:2], k=1, mode="keyword")
    assert report.recall == 1.0
    assert report.mrr == 1.0
    assert report.results[0].retrieved[0].source == "lisbon.txt"
    assert report.results[0].retrieved[0].relevant


def test_evaluate_defaults_to_the_pipeline_settings(indexed):
    report = evaluate(indexed, CASES)
    assert report.k == indexed.top_k
    assert report.mode == indexed.retrieval_mode


def test_evaluate_reports_reranking(indexed):
    class ReverseReranker:
        def score(self, question, texts):
            return [-float(i) for i in range(len(texts))]

    indexed.reranker = ReverseReranker()
    assert evaluate(indexed, CASES, k=1).reranked


def test_evaluate_report_is_json_serializable(indexed):
    data = json.loads(evaluate(indexed, CASES, k=2).model_dump_json())
    assert set(data) >= {"k", "mode", "recall", "mrr", "hit_rate", "results"}


def test_evaluate_rejects_bad_arguments(indexed):
    with pytest.raises(ValueError, match="cases"):
        evaluate(indexed, [])
    with pytest.raises(ValueError, match="k must be"):
        evaluate(indexed, CASES, k=0)
    with pytest.raises(ValueError, match="mode"):
        evaluate(indexed, CASES, mode="fuzzy")
