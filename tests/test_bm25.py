import math
import threading

import pytest
from conftest import BERLIN, LISBON

from citerag.bm25 import BM25Index, tokenize
from citerag.models import Chunk


def _chunk(cid: str, text: str, doc_id: str | None = None) -> Chunk:
    return Chunk(id=cid, doc_id=doc_id or cid.split(":")[0], source=f"{cid}.txt", text=text)


@pytest.fixture
def index() -> BM25Index:
    idx = BM25Index()
    idx.add(
        [
            _chunk("lisbon:0", LISBON),
            _chunk("berlin:0", BERLIN),
            _chunk("errors:0", "The worker crashed with error ERR_DISK_FULL after the upgrade."),
            _chunk("errors:1", "Restarting the worker fixed the timeout on the second node."),
        ]
    )
    return idx


def test_tokenize_folds_case_and_accents():
    assert tokenize("Ação, CAFÉ e Straße!") == ["acao", "cafe", "e", "strasse"]


def test_rare_exact_term_ranks_first(index):
    hits = index.search("what does err_disk_full mean", k=3)
    assert hits[0].chunk.id == "errors:0"


def test_scores_are_descending_and_positive(index):
    hits = index.search("worker capital", k=10)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_k_limits_results(index):
    assert len(index.search("the", k=2)) == 2
    assert index.search("the", k=0) == []


def test_doc_ids_filter(index):
    hits = index.search("capital", k=5, doc_ids=["berlin"])
    assert [h.chunk.doc_id for h in hits] == ["berlin"]


def test_no_matches_and_empty_index():
    assert BM25Index().search("anything", k=3) == []
    idx = BM25Index()
    idx.add([_chunk("a:0", "alpha beta")])
    assert idx.search("gamma", k=3) == []
    assert idx.search("", k=3) == []


def test_repeated_query_terms_count_once(index):
    once = index.search("worker", k=5)
    twice = index.search("worker worker", k=5)
    assert [(h.chunk.id, h.score) for h in once] == [(h.chunk.id, h.score) for h in twice]


def test_delete_matches_fresh_index(index):
    assert index.delete("errors") is True
    assert index.delete("errors") is False
    assert len(index) == 2

    fresh = BM25Index()
    fresh.add([_chunk("lisbon:0", LISBON), _chunk("berlin:0", BERLIN)])
    for query in ["capital of germany", "worker", "web summit trams"]:
        got = [(h.chunk.id, round(h.score, 9)) for h in index.search(query, k=5)]
        want = [(h.chunk.id, round(h.score, 9)) for h in fresh.search(query, k=5)]
        assert got == want


def test_readding_chunk_replaces_it(index):
    index.add([_chunk("errors:0", "Completely different words now.")])
    assert len(index) == 4
    assert index.search("err_disk_full", k=3) == []
    assert index.search("completely different", k=1)[0].chunk.id == "errors:0"


def test_known_score():
    idx = BM25Index(k1=1.2, b=0.75)
    idx.add([_chunk("a:0", "apple apple banana"), _chunk("b:0", "banana cherry")])
    # "apple" appears in 1 of 2 chunks; chunk a has tf=2, length 3, average length 2.5.
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    norm = 1 - 0.75 + 0.75 * 3 / 2.5
    expected = idf * 2 * 2.2 / (2 + 1.2 * norm)
    assert idx.search("apple", k=1)[0].score == pytest.approx(expected)


def test_ties_are_deterministic():
    idx = BM25Index()
    idx.add([_chunk("b:0", "same text"), _chunk("a:0", "same text")])
    assert [h.chunk.id for h in idx.search("same", k=2)] == ["a:0", "b:0"]


def test_tokenize_splits_cjk_into_characters():
    assert tokenize("東京は日本の首都です") == list("東京は日本の首都です")
    assert tokenize("Tokyo東京2024") == ["tokyo", "東", "京", "2024"]


def test_cjk_word_matches_inside_unspaced_text():
    idx = BM25Index()
    idx.add([_chunk("ja:0", "東京は日本の首都です"), _chunk("en:0", "Paris is in France")])
    assert idx.search("東京", k=2)[0].chunk.id == "ja:0"


def test_tokenize_keeps_vowel_signs_in_non_latin_scripts():
    assert tokenize("कम") != tokenize("कुम")
    assert tokenize("Ελλάδα") == ["ελλαδα"]


def test_moving_chunk_to_another_doc_updates_doc_index():
    idx = BM25Index()
    idx.add([_chunk("x:0", "alpha", doc_id="old")])
    idx.add([_chunk("x:0", "alpha", doc_id="new")])
    assert idx.delete("old") is False
    assert idx.delete("new") is True
    assert len(idx) == 0
    assert idx.search("alpha", k=1) == []


@pytest.mark.parametrize(
    "k1, b",
    [(-0.1, 0.75), (1.5, -0.1), (1.5, 1.1), (math.nan, 0.75), (math.inf, 0.75), (1.5, math.nan)],
)
def test_rejects_invalid_parameters(k1, b):
    with pytest.raises(ValueError):
        BM25Index(k1=k1, b=b)


def test_concurrent_adds_are_safe():
    idx = BM25Index()

    def worker(n: int) -> None:
        idx.add([_chunk(f"d{n}:{i}", f"token{n} shared text {i}") for i in range(50)])

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(idx) == 400
    assert len(idx.search("shared", k=1000)) == 400


def test_combining_marks_remain_in_words_and_kana():
    assert tokenize("किताब") == ["किताब"]
    assert tokenize("がか") == ["が", "か"]
    assert tokenize("が") != tokenize("か")
    assert len(tokenize("กำลัง")) == 1
    assert tokenize("กำลัง") != tokenize("กาลง")
    assert tokenize("CAFÉ Ελλάδα Й") == ["cafe", "ελλαδα", "и"]


def test_combining_marks_affect_keyword_search():
    idx = BM25Index()
    idx.add(
        [
            _chunk("book:0", "किताब"),
            _chunk("other:0", "किता"),
            _chunk("voiced:0", "が"),
            _chunk("plain:0", "か"),
        ]
    )
    assert [h.chunk.id for h in idx.search("किताब", k=10)] == ["book:0"]
    assert [h.chunk.id for h in idx.search("が", k=10)] == ["voiced:0"]
