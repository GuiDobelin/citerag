"""Rank fusion: merge rankings from different retrievers without calibrating their scores."""

from collections.abc import Sequence

from citerag.models import Hit

DEFAULT_RRF_K = 60


def rrf_fuse(
    rankings: Sequence[Sequence[Hit]], k: int = DEFAULT_RRF_K, limit: int | None = None
) -> list[Hit]:
    """Merge best-first rankings with Reciprocal Rank Fusion.

    Each chunk scores `sum(1 / (k + rank))` over the rankings it appears in (ranks start at
    1), so only positions matter, never the retrievers' raw scores. The returned hits carry
    the fused score. Ties break on the best single rank, then on chunk id, so the order is
    deterministic.
    """
    if k < 0:
        raise ValueError("RRF k must be >= 0.")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    hits: dict[str, Hit] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, hit in enumerate(ranking, start=1):
            cid = hit.chunk.id
            if cid in seen:  # a retriever listing a chunk twice must not count it twice
                continue
            seen.add(cid)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            best_rank[cid] = min(best_rank.get(cid, rank), rank)
            hits.setdefault(cid, hit)
    order = sorted(scores, key=lambda cid: (-scores[cid], best_rank[cid], cid))
    if limit is not None:
        order = order[: max(limit, 0)]
    return [Hit(chunk=hits[cid].chunk, score=scores[cid]) for cid in order]
