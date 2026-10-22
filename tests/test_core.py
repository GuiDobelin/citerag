import numpy as np
import pytest
from conftest import BERLIN, LISBON

from citerag import loaders
from citerag.chunking import split_text
from citerag.embeddings import HashEmbedder
from citerag.models import Chunk
from citerag.store import MemoryStore, QdrantStore


def test_split_text_respects_size_and_overlaps():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = split_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0][-20:].split()[-1] in chunks[1]


def test_split_text_short_and_empty():
    assert split_text("hello", 100, 10) == ["hello"]
    assert split_text("   ", 100, 10) == []
    with pytest.raises(ValueError):
        split_text("x", 10, 10)


def test_html_loader_strips_tags_and_scripts():
    html = b"<html><script>evil()</script><body><h1>Title</h1><p>Body text</p></body></html>"
    [(page, text)] = loaders.load("page.html", html)
    assert page is None
    assert "Title" in text and "Body text" in text and "evil" not in text


def test_loader_rejects_unknown_type():
    with pytest.raises(loaders.UnsupportedFileError):
        loaders.load("image.png", b"\x89PNG")


def test_hash_embedder_is_normalised_and_similarity_makes_sense():
    e = HashEmbedder()
    v = e.embed(["capital of Portugal", LISBON, BERLIN])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert v[0] @ v[1] > v[0] @ v[2]


def _chunks():
    return [
        Chunk(id="a:0", doc_id="a", source="lisbon.txt", text=LISBON),
        Chunk(id="b:0", doc_id="b", source="berlin.txt", text=BERLIN),
    ]


@pytest.mark.parametrize("kind", ["memory", "qdrant"])
def test_store_add_search_filter_delete(kind, tmp_path):
    e = HashEmbedder()
    store = MemoryStore(tmp_path) if kind == "memory" else QdrantStore(":memory:", "test", e.dim)
    chunks = _chunks()
    store.add(chunks, e.embed([c.text for c in chunks]))

    q = e.embed(["capital of Germany"])[0]
    assert store.search(q, 1)[0].chunk.doc_id == "b"
    assert [h.chunk.doc_id for h in store.search(q, 5, doc_ids=["a"])] == ["a"]
    assert {d.doc_id for d in store.documents()} == {"a", "b"}

    assert store.delete("a") is True
    assert store.delete("a") is False
    assert [d.doc_id for d in store.documents()] == ["b"]


def test_memory_store_persists_between_instances(tmp_path):
    e = HashEmbedder()
    chunks = _chunks()
    MemoryStore(tmp_path).add(chunks, e.embed([c.text for c in chunks]))
    reloaded = MemoryStore(tmp_path)
    assert len(reloaded.documents()) == 2
    assert reloaded.search(e.embed(["Lisbon trams"])[0], 1)[0].chunk.doc_id == "a"
