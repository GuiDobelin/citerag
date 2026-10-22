"""Embedding backends. All return L2-normalised float32 vectors."""

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0, 1, norms)).astype(np.float32)


class HashEmbedder:
    """Dependency-free bag-of-words embedder using feature hashing.

    Good enough for tests and keyword-style lookups; use FastEmbedEmbedder for real semantics.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                h = int.from_bytes(digest, "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalise(out)


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed (no API key, runs on CPU)."""

    def __init__(self, model_name: str) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "fastembed is not installed. Run `pip install citerag[local]` "
                "or set CITERAG_EMBEDDER=hash."
            ) from exc
        self._model = TextEmbedding(model_name=model_name)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.array(list(self._model.embed(texts)), dtype=np.float32))
