import pytest

from citerag.fusion import rrf_fuse
from citerag.models import Chunk, Hit


def _hits(*ids: str, score: float = 1.0) -> list[Hit]:
    return [
        Hit(chunk=Chunk(id=cid, doc_id=cid.split(":")[0], source="s.txt", text=cid), score=score)
        for cid in ids
    ]


def _ids(hits: list[Hit]) -> list[str]:
    return [h.chunk.id for h in hits]


def test_rrf_uses_ranks_not_raw_scores():
    # The second list's huge scores must not dominate: only positions count.
    fused = rrf_fuse([_hits("a:0", "b:0", score=0.1), _hits("b:0", "c:0", score=1000.0)])
    assert _ids(fused) == ["b:0", "a:0", "c:0"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)


def test_rrf_respects_k_and_limit():
    fused = rrf_fuse([_hits("a:0", "b:0", "c:0")], k=0, limit=2)
    assert _ids(fused) == ["a:0", "b:0"]
    assert [h.score for h in fused] == pytest.approx([1.0, 0.5])
    assert rrf_fuse([_hits("a:0")], limit=0) == []


def test_rrf_ties_are_deterministic():
    # a and x are both first in one list: same score and same best rank, so id decides.
    assert _ids(rrf_fuse([_hits("x:0"), _hits("a:0")])) == ["a:0", "x:0"]
    # Same total, but b has the better single rank.
    fused = rrf_fuse([_hits("b:0", "z:0", "a:0"), _hits("a:0", "q:0", "b:0")], k=1)
    assert _ids(fused)[:2] == ["a:0", "b:0"]


def test_rrf_counts_a_duplicate_within_one_ranking_once():
    fused = rrf_fuse([_hits("a:0", "a:0"), _hits("b:0")])
    assert fused[0].score == pytest.approx(fused[1].score)


def test_rrf_edge_cases():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []
    with pytest.raises(ValueError):
        rrf_fuse([_hits("a:0")], k=-1)
