import numpy as np
import pytest
from conftest import BERLIN, LISBON

from citerag.config import Settings
from citerag.embeddings import HashEmbedder
from citerag.llm import ExtractiveGenerator
from citerag.models import Chunk
from citerag.pipeline import RAG
from citerag.store import MemoryStore

RARE = "Error code ZX-4471 means the flux capacitor overheated."


class BlindEmbedder:
    """Puts every text at the same point, so vector ranking ignores the words entirely."""

    dim = 4

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))


def _rag(tmp_path, mode="hybrid", embedder=None, store=None) -> RAG:
    return RAG(
        embedder=embedder or HashEmbedder(),
        store=store or MemoryStore(tmp_path / "store"),
        generator=ExtractiveGenerator(),
        chunk_size=200,
        chunk_overlap=20,
        top_k=1,
        retrieval_mode=mode,
    )


def _ingest_corpus(rag: RAG) -> None:
    # The rare document is ingested last; with identical vectors the vector store keeps
    # insertion order, so a vector-only ranker returns a filler document first.
    for n in range(5):
        rag.ingest(f"filler{n}.txt", f"Filler document number {n} about trams.".encode())
    rag.ingest("lisbon.txt", LISBON.encode())
    rag.ingest("rare.txt", RARE.encode())


def test_hybrid_finds_a_rare_exact_term_the_vector_ranker_misses(tmp_path):
    question = "What does ZX-4471 mean?"
    vector = _rag(tmp_path / "v", "vector", BlindEmbedder())
    _ingest_corpus(vector)
    assert vector.retrieve(question)[0].chunk.source != "rare.txt"

    hybrid = _rag(tmp_path / "h", "hybrid", BlindEmbedder())
    _ingest_corpus(hybrid)
    assert hybrid.retrieve(question)[0].chunk.source == "rare.txt"
    assert hybrid.ask(question).citations[0].source == "rare.txt"


def test_vector_mode_reproduces_plain_vector_search(tmp_path):
    rag = _rag(tmp_path, "vector")
    rag.ingest("lisbon.txt", LISBON.encode())
    rag.ingest("berlin.md", BERLIN.encode())
    question = "Which city hosts Web Summit?"
    expected = rag.store.search(rag.embedder.embed([question])[0], 3)
    assert rag.retrieve(question, 3) == expected
    assert rag._keyword_index is None  # vector mode never builds the keyword index


def test_keyword_mode_uses_bm25_only(tmp_path):
    rag = _rag(tmp_path, "keyword", BlindEmbedder())
    _ingest_corpus(rag)
    assert rag.retrieve("ZX-4471")[0].chunk.source == "rare.txt"
    assert rag.retrieve("no shared words") == []


@pytest.mark.parametrize("mode", ["keyword", "hybrid"])
def test_deleting_a_document_removes_it_from_keyword_results(tmp_path, mode):
    rag = _rag(tmp_path, mode, BlindEmbedder())
    _ingest_corpus(rag)
    rare_id = next(d.doc_id for d in rag.documents() if d.source == "rare.txt")
    assert rag.delete(rare_id) is True
    assert rag.delete(rare_id) is False
    hits = rag.retrieve("ZX-4471", 10)
    assert all(h.chunk.doc_id != rare_id for h in hits)


def test_reingest_replaces_keyword_entries(tmp_path):
    rag = _rag(tmp_path, "keyword")
    rag.ingest("rare.txt", RARE.encode())
    rag.ingest("rare.txt", RARE.encode())
    assert len(rag._keywords()) == 1


def test_doc_ids_filter_applies_to_both_retrievers(tmp_path):
    rag = _rag(tmp_path, "hybrid", BlindEmbedder())
    _ingest_corpus(rag)
    lisbon_id = next(d.doc_id for d in rag.documents() if d.source == "lisbon.txt")
    hits = rag.retrieve("ZX-4471 trams", 10, doc_ids=[lisbon_id])
    assert hits and {h.chunk.doc_id for h in hits} == {lisbon_id}


def test_keyword_index_is_rebuilt_from_a_persisted_store(tmp_path):
    first = _rag(tmp_path, "vector", BlindEmbedder())
    _ingest_corpus(first)
    # A new process: the store loads from disk and the keyword index is rebuilt from it.
    restarted = _rag(tmp_path, "hybrid", BlindEmbedder())
    assert len(restarted._keyword_index) == len(restarted.store.chunks())
    assert restarted.retrieve("ZX-4471")[0].chunk.source == "rare.txt"


def test_keyword_index_is_rebuilt_from_qdrant():
    pytest.importorskip("qdrant_client")
    from citerag.store import QdrantStore

    store = QdrantStore(":memory:", "test", BlindEmbedder.dim)
    chunk = Chunk(id="r:0", doc_id="r", source="rare.txt", text=RARE)
    store.add([chunk], BlindEmbedder().embed([RARE]))
    assert store.chunks() == [chunk]
    rag = RAG(BlindEmbedder(), store, ExtractiveGenerator(), top_k=1, retrieval_mode="keyword")
    assert rag.retrieve("ZX-4471")[0].chunk == chunk


def test_retrieval_mode_is_validated(tmp_path):
    with pytest.raises(ValueError):
        _rag(tmp_path, "semantic")


def test_settings_default_to_hybrid(monkeypatch, tmp_path):
    monkeypatch.delenv("CITERAG_RETRIEVAL_MODE", raising=False)
    assert Settings(_env_file=None).retrieval_mode == "hybrid"
    monkeypatch.setenv("CITERAG_RETRIEVAL_MODE", "vector")
    monkeypatch.setenv("CITERAG_EMBEDDER", "hash")
    monkeypatch.setenv("CITERAG_GENERATOR", "extractive")
    monkeypatch.setenv("CITERAG_DATA_DIR", str(tmp_path))
    rag = RAG.from_settings()
    assert rag.retrieval_mode == "vector"
    assert rag.rrf_k == 60
