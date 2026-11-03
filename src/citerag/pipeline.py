import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import get_args

from citerag import loaders
from citerag.bm25 import BM25Index
from citerag.chunking import split_text
from citerag.config import Settings
from citerag.embeddings import Embedder, FastEmbedEmbedder, HashEmbedder
from citerag.fusion import DEFAULT_RRF_K, rrf_fuse
from citerag.llm import ClaudeGenerator, ExtractiveGenerator, Generator
from citerag.models import Answer, Chunk, DocumentInfo, Hit, RetrievalMode, StreamEvent
from citerag.store import MemoryStore, QdrantStore, VectorStore

RETRIEVAL_MODES: tuple[RetrievalMode, ...] = get_args(RetrievalMode)
# In hybrid mode each retriever returns this many times `k` candidates before fusion, so a
# chunk ranked just below `k` by both retrievers can still win.
CANDIDATE_FACTOR = 4


class RAG:
    """Ingest documents, retrieve relevant chunks, and generate cited answers.

    Retrieval ranks chunks by vector similarity, by BM25 keyword relevance, or by both fused
    with Reciprocal Rank Fusion (`hybrid`). The keyword index lives in memory: it is built
    from the store on the first keyword or hybrid query and kept in sync by `ingest` and
    `delete`, so documents should be removed through `RAG.delete`, not on the store.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        generator: Generator,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        top_k: int = 5,
        retrieval_mode: RetrievalMode = "hybrid",
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval_mode must be one of {', '.join(RETRIEVAL_MODES)}.")
        if rrf_k < 0:
            raise ValueError("rrf_k must be >= 0.")
        self.embedder = embedder
        self.store = store
        self.generator = generator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode
        self.rrf_k = rrf_k
        # Guards the keyword index together with the store writes it mirrors.
        self._lock = threading.RLock()
        self._keyword_index: BM25Index | None = None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RAG":
        s = settings or Settings()
        embedder: Embedder = (
            HashEmbedder() if s.embedder == "hash" else FastEmbedEmbedder(s.embedding_model)
        )
        if s.store == "qdrant":
            store: VectorStore = QdrantStore(
                s.qdrant_url or ":memory:", s.qdrant_collection, embedder.dim
            )
        else:
            store = MemoryStore(s.data_dir)
        generator: Generator = (
            ExtractiveGenerator()
            if s.generator == "extractive"
            else ClaudeGenerator(s.claude_model, s.claude_effort, s.max_answer_tokens)
        )
        return cls(
            embedder,
            store,
            generator,
            s.chunk_size,
            s.chunk_overlap,
            s.top_k,
            retrieval_mode=s.retrieval_mode,
            rrf_k=s.rrf_k,
        )

    def ingest(self, filename: str, data: bytes) -> DocumentInfo:
        """Index a file. Re-ingesting identical content replaces the previous copy."""
        doc_id = hashlib.sha256(data).hexdigest()[:16]
        chunks = [
            Chunk(id=f"{doc_id}:{n}", doc_id=doc_id, source=filename, page=page, text=text)
            for n, (page, text) in enumerate(
                (page, piece)
                for page, page_text in loaders.load(filename, data)
                for piece in split_text(page_text, self.chunk_size, self.chunk_overlap)
            )
        ]
        if not chunks:
            raise ValueError(f"No extractable text found in '{filename}'.")
        vectors = self.embedder.embed([c.text for c in chunks])
        with self._lock, self._resync_on_error():
            self.store.delete(doc_id)
            self.store.add(chunks, vectors)
            if self._keyword_index is not None:
                self._keyword_index.delete(doc_id)
                self._keyword_index.add(chunks)
        return DocumentInfo(doc_id=doc_id, source=filename, chunks=len(chunks))

    def documents(self) -> list[DocumentInfo]:
        return self.store.documents()

    def delete(self, doc_id: str) -> bool:
        """Remove a document from the store and the keyword index. False if it was unknown."""
        with self._lock, self._resync_on_error():
            deleted = self.store.delete(doc_id)
            if self._keyword_index is not None:
                self._keyword_index.delete(doc_id)
            return deleted

    def retrieve(
        self, question: str, k: int | None = None, doc_ids: list[str] | None = None
    ) -> list[Hit]:
        k = self.top_k if k is None else k
        if k < 1:
            raise ValueError("k must be >= 1.")
        if self.retrieval_mode == "keyword":
            return self._keywords().search(question, k, doc_ids)
        vector = self.embedder.embed([question])[0]
        if self.retrieval_mode == "vector":
            return self.store.search(vector, k, doc_ids)
        pool = k * CANDIDATE_FACTOR
        rankings = [
            self.store.search(vector, pool, doc_ids),
            self._keywords().search(question, pool, doc_ids),
        ]
        return rrf_fuse(rankings, k=self.rrf_k, limit=k)

    def ask(self, question: str, k: int | None = None, doc_ids: list[str] | None = None) -> Answer:
        return self.generator.generate(question, self.retrieve(question, k, doc_ids))

    def ask_stream(
        self, question: str, k: int | None = None, doc_ids: list[str] | None = None
    ) -> Iterator[StreamEvent]:
        """Like `ask`, but yields the answer as it is generated. The last event is `Done`.

        Retrieval runs before this returns, so its errors surface here rather than mid-stream.
        """
        hits = self.retrieve(question, k, doc_ids)
        return self.generator.stream(question, hits)

    def _keywords(self) -> BM25Index:
        """The keyword index, built from every chunk in the store the first time it is needed."""
        with self._lock:
            if self._keyword_index is None:
                index = BM25Index()
                index.add(self.store.chunks())
                self._keyword_index = index
            return self._keyword_index

    @contextmanager
    def _resync_on_error(self) -> Iterator[None]:
        """If a store write fails midway, drop the keyword index so it is rebuilt from the store
        on the next query instead of drifting from it."""
        try:
            yield
        except BaseException:
            self._keyword_index = None
            raise
