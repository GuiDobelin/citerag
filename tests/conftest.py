from concurrent.futures import Executor, Future

import httpx
import pytest
from fastapi.testclient import TestClient

from citerag.api import create_app, get_job_manager, get_rag
from citerag.embeddings import HashEmbedder
from citerag.fetch import Fetcher
from citerag.jobs import JobManager
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


def html(text: str) -> httpx.Response:
    """A 200 HTML response with `text` in a paragraph."""
    return httpx.Response(
        200, headers={"content-type": "text/html; charset=utf-8"}, text=f"<p>{text}</p>"
    )


def make_fetcher(handler, dns=None, **kwargs) -> Fetcher:
    """A `Fetcher` that never touches the network: `handler` answers every request."""
    kwargs.setdefault("max_bytes", 1024 * 1024)
    return Fetcher(resolver=dns or FakeDNS(), transport=httpx.MockTransport(handler), **kwargs)


class InlineExecutor(Executor):
    """Runs each job as soon as it is submitted, so tests need no threads."""

    def submit(self, fn, /, *args, **kwargs):
        future: Future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


class HeldExecutor(Executor):
    """Keeps submitted jobs until `run_all`, to observe the `queued` state."""

    def __init__(self) -> None:
        self.pending: list = []

    def submit(self, fn, /, *args, **kwargs):
        self.pending.append((fn, args))
        return Future()

    def run_all(self) -> None:
        while self.pending:
            fn, args = self.pending.pop(0)
            fn(*args)

    def shutdown(self, wait=True, *, cancel_futures=False):
        if cancel_futures:
            self.pending.clear()


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


@pytest.fixture
def job_manager():
    """A real `JobManager` with worker threads."""
    manager = JobManager()
    yield manager
    manager.shutdown()


@pytest.fixture
def client(rag, job_manager):
    """A test client for the API, backed by the `rag` and `job_manager` fixtures."""
    app = create_app()
    app.dependency_overrides[get_rag] = lambda: rag
    app.dependency_overrides[get_job_manager] = lambda: job_manager
    return TestClient(app)
