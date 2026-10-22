import pytest

from citerag.embeddings import HashEmbedder
from citerag.llm import ExtractiveGenerator
from citerag.pipeline import RAG
from citerag.store import MemoryStore

LISBON = (
    "Lisbon is the capital of Portugal. It sits on the Tagus river and is known for its trams.\n\n"
    "The city hosts Web Summit, one of the largest technology conferences in Europe."
)
BERLIN = (
    "Berlin is the capital of Germany. Its startup scene is strong in fintech and mobility, "
    "and many companies there hire Python and TypeScript developers."
)


@pytest.fixture
def rag(tmp_path) -> RAG:
    return RAG(
        embedder=HashEmbedder(),
        store=MemoryStore(tmp_path / "store"),
        generator=ExtractiveGenerator(),
        chunk_size=200,
        chunk_overlap=20,
        top_k=3,
    )
