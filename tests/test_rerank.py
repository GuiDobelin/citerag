import math
import sys

import pytest
from conftest import BERLIN, LISBON

from citerag import pipeline
from citerag.config import Settings
from citerag.embeddings import HashEmbedder
from citerag.llm import ExtractiveGenerator
from citerag.models import Chunk, Done, Hit
from citerag.pipeline import RAG
from citerag.rerank import FastEmbedReranker, rerank
from citerag.store import MemoryStore


class KeywordReranker:
    """Scores a passage 1.0 if it mentions `word`, else 0.0, and records every call."""

    def __init__(self, word: str) -> None:
        self.word = word
        self.calls: list[list[str]] = []

    def score(self, question, passages):
        self.calls.append(list(passages))
        return [1.0 if self.word in p else 0.0 for p in passages]


class FixedReranker:
    def __init__(self, scores):
        self.scores = scores

    def score(self, question, passages):
        return list(self.scores)


def _hits(*texts: str) -> list[Hit]:
    return [
        Hit(chunk=Chunk(id=f"d:{n}", doc_id="d", source="d.txt", text=t), score=1.0 - n / 10)
        for n, t in enumerate(texts)
    ]


def _texts(hits: list[Hit]) -> list[str]:
    return [h.chunk.text for h in hits]


def _rag(tmp_path, reranker=None, **kwargs) -> RAG:
    rag = RAG(
        embedder=HashEmbedder(),
        store=MemoryStore(tmp_path / "store"),
        generator=ExtractiveGenerator(),
        chunk_size=200,
        chunk_overlap=20,
        top_k=1,
        reranker=reranker,
        **kwargs,
    )
    rag.ingest("lisbon.txt", LISBON.encode())
    rag.ingest("berlin.md", BERLIN.encode())
    for n in range(5):
        rag.ingest(f"filler{n}.txt", f"Filler note {n}: the capital has trams.".encode())
    return rag


def test_rerank_orders_by_score_and_cuts_to_k():
    hits = _hits("a", "b", "c", "d")
    before = [h.score for h in hits]
    out = rerank(FixedReranker([0.1, 0.95, 0.5, 0.7]), "q", hits, 3)
    assert _texts(out) == ["b", "d", "c"]
    assert [h.score for h in out] == [0.95, 0.7, 0.5]
    assert [h.score for h in hits] == before  # the input hits are not mutated


def test_rerank_puts_an_infinite_score_first():
    out = rerank(FixedReranker([0.5, math.inf, -math.inf]), "q", _hits("a", "b", "c"), 3)
    assert _texts(out) == ["b", "a", "c"]


def test_rerank_keeps_first_stage_order_for_ties_and_non_finite_scores():
    hits = _hits("a", "b", "c", "d")
    out = rerank(FixedReranker([math.nan, 0.5, 0.5, -math.inf]), "q", hits, 4)
    assert _texts(out) == ["b", "c", "a", "d"]


def test_rerank_without_hits_does_not_call_the_model():
    reranker = KeywordReranker("x")
    assert rerank(reranker, "q", [], 3) == []
    assert reranker.calls == []


def test_rerank_rejects_a_wrong_number_of_scores():
    with pytest.raises(ValueError, match="2 scores for 3 passages"):
        rerank(FixedReranker([1.0, 2.0]), "q", _hits("a", "b", "c"), 3)


def test_reranker_decides_the_final_order(tmp_path):
    question = "Which city is the capital?"
    plain = _rag(tmp_path / "plain")
    assert "Berlin" not in plain.retrieve(question)[0].chunk.text

    reranker = KeywordReranker("Berlin")
    rag = _rag(tmp_path / "rr", reranker, rerank_candidates=10)
    hits = rag.retrieve(question)
    assert len(hits) == 1  # cut to top_k
    assert hits[0].chunk.source == "berlin.md"
    assert hits[0].score == 1.0
    assert len(reranker.calls) == 1
    assert 1 < len(reranker.calls[0]) <= 10


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_reranker_sees_the_first_stage_candidates_of_every_mode(tmp_path, mode):
    question = "capital trams"
    plain = _rag(tmp_path / "plain", retrieval_mode=mode)
    reranker = KeywordReranker("never-matches")
    rag = _rag(tmp_path / "rr", reranker, retrieval_mode=mode, rerank_candidates=4)
    assert reranker.calls == []
    # With all-equal scores the reranker keeps the first-stage order.
    assert _texts(rag.retrieve(question)) == _texts(plain.retrieve(question, 1))
    assert reranker.calls[0] == _texts(plain.retrieve(question, 4))


def test_reranker_gets_at_least_k_candidates(tmp_path):
    reranker = KeywordReranker("Berlin")
    rag = _rag(tmp_path, reranker, rerank_candidates=2)
    assert len(rag.retrieve("capital", k=5)) == 5
    assert len(reranker.calls[0]) == 5


def test_ask_and_ask_stream_use_the_reranker(tmp_path):
    rag = _rag(tmp_path, KeywordReranker("Berlin"))
    question = "Which city is the capital?"
    assert rag.ask(question).citations[0].source == "berlin.md"
    done = list(rag.ask_stream(question))[-1]
    assert isinstance(done, Done)
    assert done.answer.citations[0].source == "berlin.md"


def test_rag_rejects_invalid_rerank_candidates(tmp_path):
    with pytest.raises(ValueError, match="rerank_candidates"):
        RAG(HashEmbedder(), MemoryStore(tmp_path), ExtractiveGenerator(), rerank_candidates=0)


def test_settings_default_to_no_reranker(tmp_path, monkeypatch):
    monkeypatch.delenv("CITERAG_RERANKER", raising=False)
    rag = RAG.from_settings(
        Settings(embedder="hash", generator="extractive", data_dir=tmp_path, _env_file=None)
    )
    assert rag.reranker is None
    assert rag.rerank_candidates == 20


def test_settings_build_the_fastembed_reranker(tmp_path, monkeypatch):
    built = []

    class FakeFastEmbedReranker:
        def __init__(self, model_name):
            built.append(model_name)

    monkeypatch.setattr(pipeline, "FastEmbedReranker", FakeFastEmbedReranker)
    monkeypatch.setenv("CITERAG_RERANKER", "fastembed")
    monkeypatch.setenv("CITERAG_RERANK_MODEL", "BAAI/bge-reranker-base")
    monkeypatch.setenv("CITERAG_RERANK_CANDIDATES", "8")
    rag = RAG.from_settings(
        Settings(embedder="hash", generator="extractive", data_dir=tmp_path, _env_file=None)
    )
    assert isinstance(rag.reranker, FakeFastEmbedReranker)
    assert built == ["BAAI/bge-reranker-base"]
    assert rag.rerank_candidates == 8


def test_settings_reject_invalid_rerank_candidates(monkeypatch):
    monkeypatch.setenv("CITERAG_RERANK_CANDIDATES", "0")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_fastembed_reranker_explains_a_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", None)
    with pytest.raises(ImportError, match="CITERAG_RERANKER=none"):
        FastEmbedReranker()
