"""Okapi BM25 keyword index over chunks, the lexical half of hybrid search."""

import math
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable

from citerag.models import Chunk, Hit

# Chinese and Japanese are written without spaces, so each ideograph or kana is a token.
_CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_TOKEN = re.compile(rf"[{_CJK}]|[^\W{_CJK}]+")
# Accents are only stripped from Latin, Greek and Cyrillic letters; in scripts such as
# Devanagari or Thai the combining marks are vowels and change the word.
_FOLDABLE_BELOW = 0x0530


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents (so "ação" matches "acao") and split into word tokens."""
    kept: list[str] = []
    for ch in unicodedata.normalize("NFKD", text.casefold()):
        if unicodedata.combining(ch) and kept and ord(kept[-1]) < _FOLDABLE_BELOW:
            continue
        kept.append(ch)
    return _TOKEN.findall(unicodedata.normalize("NFC", "".join(kept)))


class BM25Index:
    """In-memory inverted index ranked with Okapi BM25.

    Scores are only comparable within one query, which is why hybrid search fuses rankings
    rather than raw scores.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if not (math.isfinite(k1) and k1 >= 0 and 0 <= b <= 1):
            raise ValueError("BM25 needs a finite k1 >= 0 and 0 <= b <= 1.")
        self.k1 = k1
        self.b = b
        self._lock = threading.RLock()
        self._chunks: dict[str, Chunk] = {}
        self._terms: dict[str, list[str]] = {}
        self._lengths: dict[str, int] = {}
        self._by_doc: dict[str, set[str]] = defaultdict(set)
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._total_length = 0

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: Iterable[Chunk]) -> None:
        """Index chunks. A chunk whose id is already indexed is replaced."""
        with self._lock:
            for chunk in chunks:
                self._remove(chunk.id)
                counts = Counter(tokenize(chunk.text))
                self._chunks[chunk.id] = chunk
                self._terms[chunk.id] = list(counts)
                self._by_doc[chunk.doc_id].add(chunk.id)
                self._lengths[chunk.id] = sum(counts.values())
                self._total_length += self._lengths[chunk.id]
                for term, tf in counts.items():
                    self._postings[term][chunk.id] = tf

    def delete(self, doc_id: str) -> bool:
        """Remove every chunk of a document. Returns False if it was not indexed."""
        with self._lock:
            ids = self._by_doc.pop(doc_id, set())
            for cid in list(ids):
                self._remove(cid)
            return bool(ids)

    def search(self, query: str, k: int, doc_ids: list[str] | None = None) -> list[Hit]:
        """Return up to `k` chunks that share at least one term with `query`, best first."""
        with self._lock:
            n = len(self._chunks)
            if not n or k <= 0:
                return []
            allowed = set(doc_ids) if doc_ids else None
            avg_length = self._total_length / n or 1.0
            scores: dict[str, float] = defaultdict(float)
            for term in set(tokenize(query)):
                postings = self._postings.get(term)
                if not postings:
                    continue
                idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
                for cid, tf in postings.items():
                    if allowed is not None and self._chunks[cid].doc_id not in allowed:
                        continue
                    norm = 1 - self.b + self.b * self._lengths[cid] / avg_length
                    scores[cid] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
            # Ties break on chunk id so results are deterministic.
            best = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
            return [Hit(chunk=self._chunks[cid], score=score) for cid, score in best]

    def _remove(self, chunk_id: str) -> None:
        chunk = self._chunks.pop(chunk_id, None)
        if chunk is None:
            return
        self._total_length -= self._lengths.pop(chunk_id)
        doc_chunks = self._by_doc.get(chunk.doc_id)
        if doc_chunks is not None:
            doc_chunks.discard(chunk_id)
            if not doc_chunks:
                del self._by_doc[chunk.doc_id]
        for term in self._terms.pop(chunk_id):
            postings = self._postings[term]
            postings.pop(chunk_id, None)
            if not postings:
                del self._postings[term]
