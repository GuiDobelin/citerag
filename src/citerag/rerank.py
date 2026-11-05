"""Optional second-stage rerankers that reorder retrieved chunks by reading question and text
together. They are slower than first-stage retrieval, so they only see a short candidate list."""

import math
from collections.abc import Sequence
from typing import Protocol

from citerag.models import Hit

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        """One relevance score per passage, in input order. Higher means more relevant."""
        ...


def rerank(reranker: Reranker, question: str, hits: list[Hit], k: int) -> list[Hit]:
    """Reorder `hits` by the reranker's scores and keep the best `k`.

    The returned hits carry the reranker score. Ties, and scores that are not finite numbers,
    keep the order of the first-stage ranking.
    """
    if not hits:
        return []
    scores = reranker.score(question, [hit.chunk.text for hit in hits])
    if len(scores) != len(hits):
        raise ValueError(f"Reranker returned {len(scores)} scores for {len(hits)} passages.")
    scored = [
        (s if math.isfinite(s) else -math.inf, hit) for s, hit in zip(scores, hits, strict=True)
    ]
    # sorted() is stable, so equal scores keep their first-stage order.
    ranked = sorted(scored, key=lambda pair: pair[0], reverse=True)
    return [hit.model_copy(update={"score": float(s)}) for s, hit in ranked[:k]]


class FastEmbedReranker:
    """Local ONNX cross-encoder via fastembed (no API key, runs on CPU)."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:
            raise ImportError(
                "fastembed is not installed. Run `pip install citerag[local]` "
                "or set CITERAG_RERANKER=none."
            ) from exc
        self._model = TextCrossEncoder(model_name=model_name)

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(question, list(passages))]
