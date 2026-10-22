"""Vector stores: an in-process store for single-node use and Qdrant for scaling out."""

import json
import threading
import uuid
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np

from citerag.models import Chunk, DocumentInfo, Hit


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, vector: np.ndarray, k: int, doc_ids: list[str] | None = None) -> list[Hit]: ...
    def documents(self) -> list[DocumentInfo]: ...
    def delete(self, doc_id: str) -> bool: ...


class MemoryStore:
    """Brute-force cosine search in NumPy, optionally persisted to `path`.

    Fast up to a few hundred thousand chunks; beyond that, switch to QdrantStore.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        if path and (path / "chunks.json").exists():
            self._chunks = [Chunk(**c) for c in json.loads((path / "chunks.json").read_text())]
            self._vectors = np.load(path / "vectors.npy")

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        with self._lock:
            self._chunks.extend(chunks)
            self._vectors = (
                vectors if self._vectors is None else np.vstack([self._vectors, vectors])
            )
            self._save()

    def search(self, vector: np.ndarray, k: int, doc_ids: list[str] | None = None) -> list[Hit]:
        with self._lock:
            if self._vectors is None or not self._chunks:
                return []
            scores = self._vectors @ vector
            if doc_ids:
                allowed = np.array([c.doc_id in doc_ids for c in self._chunks])
                scores = np.where(allowed, scores, -np.inf)
            top = np.argsort(-scores)[:k]
            return [
                Hit(chunk=self._chunks[i], score=float(scores[i]))
                for i in top
                if np.isfinite(scores[i])
            ]

    def documents(self) -> list[DocumentInfo]:
        with self._lock:
            counts = Counter(c.doc_id for c in self._chunks)
            sources = {c.doc_id: c.source for c in self._chunks}
            return [DocumentInfo(doc_id=d, source=sources[d], chunks=n) for d, n in counts.items()]

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            keep = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
            if len(keep) == len(self._chunks):
                return False
            self._chunks = [self._chunks[i] for i in keep]
            self._vectors = self._vectors[keep] if keep else None
            self._save()
            return True

    def _save(self) -> None:
        if not self._path:
            return
        self._path.mkdir(parents=True, exist_ok=True)
        (self._path / "chunks.json").write_text(
            json.dumps([c.model_dump() for c in self._chunks], ensure_ascii=False)
        )
        if self._vectors is not None:
            np.save(self._path / "vectors.npy", self._vectors)
        else:
            (self._path / "vectors.npy").unlink(missing_ok=True)


class QdrantStore:
    """Qdrant-backed store. Pass url=":memory:" for an embedded instance (handy in tests)."""

    def __init__(self, url: str, collection: str, dim: int) -> None:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is not installed. Run `pip install citerag[qdrant]`."
            ) from exc
        self._models = models
        self._client = QdrantClient(location=url) if url == ":memory:" else QdrantClient(url=url)
        self._collection = collection
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            if url != ":memory:":  # embedded Qdrant ignores payload indexes
                self._client.create_payload_index(
                    collection, "doc_id", models.PayloadSchemaType.KEYWORD
                )

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        points = [
            self._models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.id)),
                vector=v.tolist(),
                payload=c.model_dump(),
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(self._collection, points=points)

    def search(self, vector: np.ndarray, k: int, doc_ids: list[str] | None = None) -> list[Hit]:
        m = self._models
        query_filter = (
            m.Filter(must=[m.FieldCondition(key="doc_id", match=m.MatchAny(any=doc_ids))])
            if doc_ids
            else None
        )
        result = self._client.query_points(
            self._collection, query=vector.tolist(), limit=k, query_filter=query_filter
        )
        return [Hit(chunk=Chunk(**p.payload), score=p.score) for p in result.points]

    def documents(self) -> list[DocumentInfo]:
        counts: Counter[str] = Counter()
        sources: dict[str, str] = {}
        offset = None
        while True:
            points, offset = self._client.scroll(
                self._collection, limit=1000, offset=offset, with_payload=["doc_id", "source"]
            )
            for p in points:
                counts[p.payload["doc_id"]] += 1
                sources[p.payload["doc_id"]] = p.payload["source"]
            if offset is None:
                break
        return [DocumentInfo(doc_id=d, source=sources[d], chunks=n) for d, n in counts.items()]

    def delete(self, doc_id: str) -> bool:
        m = self._models
        doc_filter = m.Filter(
            must=[m.FieldCondition(key="doc_id", match=m.MatchValue(value=doc_id))]
        )
        if not self._client.count(self._collection, count_filter=doc_filter).count:
            return False
        self._client.delete(self._collection, points_selector=m.FilterSelector(filter=doc_filter))
        return True
