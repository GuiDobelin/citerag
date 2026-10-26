import hashlib
from collections.abc import Iterator

from citerag import loaders
from citerag.chunking import split_text
from citerag.config import Settings
from citerag.embeddings import Embedder, FastEmbedEmbedder, HashEmbedder
from citerag.llm import ClaudeGenerator, ExtractiveGenerator, Generator
from citerag.models import Answer, Chunk, DocumentInfo, Hit, StreamEvent
from citerag.store import MemoryStore, QdrantStore, VectorStore


class RAG:
    """Ingest documents, retrieve relevant chunks, and generate cited answers."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        generator: Generator,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        top_k: int = 5,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.generator = generator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

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
        return cls(embedder, store, generator, s.chunk_size, s.chunk_overlap, s.top_k)

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
        self.store.delete(doc_id)
        self.store.add(chunks, self.embedder.embed([c.text for c in chunks]))
        return DocumentInfo(doc_id=doc_id, source=filename, chunks=len(chunks))

    def retrieve(
        self, question: str, k: int | None = None, doc_ids: list[str] | None = None
    ) -> list[Hit]:
        vector = self.embedder.embed([question])[0]
        return self.store.search(vector, k or self.top_k, doc_ids)

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
