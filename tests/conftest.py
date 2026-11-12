import httpx
import pytest

from citerag.embeddings import HashEmbedder
from citerag.fetch import Fetcher
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
PUBLIC_IP = "93.184.215.14"


class FakeDNS:
    """Resolver with fixed answers; unknown hosts resolve to a public address."""

    def __init__(self, records: dict[str, list[str]] | None = None) -> None:
        self.records = records or {}
        self.lookups: list[tuple[str, int]] = []

    def __call__(self, host: str, port: int) -> list[str]:
        self.lookups.append((host, port))
        return self.records.get(host, [PUBLIC_IP])


def make_fetcher(handler, dns=None, **kwargs) -> Fetcher:
    """A `Fetcher` that never touches the network: `handler` answers every request."""
    kwargs.setdefault("max_bytes", 1024 * 1024)
    return Fetcher(resolver=dns or FakeDNS(), transport=httpx.MockTransport(handler), **kwargs)


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
