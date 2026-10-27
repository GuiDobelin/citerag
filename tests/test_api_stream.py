import json

import pytest
from conftest import BERLIN, LISBON
from fastapi.testclient import TestClient

from citerag.api import create_app, get_rag
from citerag.llm import GenerationError
from citerag.models import TextDelta


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs, insisting on one `data:` line per frame."""
    frames = []
    for raw in body.split("\n\n"):
        if not raw:
            continue
        lines = raw.split("\n")
        assert len(lines) == 2, f"unexpected frame: {raw!r}"
        event, data = lines
        assert event.startswith("event: ") and data.startswith("data: ")
        frames.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return frames


@pytest.fixture
def client(rag):
    rag.ingest("lisbon.txt", LISBON.encode())
    rag.ingest("berlin.md", BERLIN.encode())
    app = create_app()
    app.dependency_overrides[get_rag] = lambda: rag
    return TestClient(app)


def test_stream_sends_deltas_citations_and_done(client):
    body = {"question": "Which city hosts Web Summit?"}
    r = client.post("/query/stream", json=body)

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    assert r.headers["x-accel-buffering"] == "no"

    frames = parse_sse(r.text)
    kinds = [kind for kind, _ in frames]
    assert kinds[-1] == "done" and kinds.count("done") == 1
    assert {"delta", "citation"} <= set(kinds)

    done = frames[-1][1]
    assert done == client.post("/query", json=body).json()
    assert "".join(d["text"] for k, d in frames if k == "delta") == done["answer"]
    cites = [d for k, d in frames if k == "citation"]
    assert [c["index"] for c in cites] == list(range(1, len(done["citations"]) + 1))
    assert [c["citation"] for c in cites] == done["citations"]


def test_stream_respects_doc_ids_and_top_k(client, rag):
    berlin = next(d.doc_id for d in rag.store.documents() if d.source == "berlin.md")
    r = client.post("/query/stream", json={"question": "capital", "top_k": 1, "doc_ids": [berlin]})
    done = parse_sse(r.text)[-1][1]
    assert [c["doc_id"] for c in done["citations"]] == [berlin]


class FailingGenerator:
    name = "failing"

    def __init__(self, error: Exception) -> None:
        self.error = error

    def generate(self, question, hits):
        raise self.error

    def stream(self, question, hits):
        yield TextDelta(text="Partial ")
        raise self.error


def test_stream_reports_generation_errors_as_error_event(client, rag):
    rag.generator = FailingGenerator(GenerationError("Claude rate limit reached, retry later."))
    r = client.post("/query/stream", json={"question": "Where is Web Summit?"})
    assert r.status_code == 200
    assert parse_sse(r.text) == [
        ("delta", {"text": "Partial "}),
        ("error", {"detail": "Claude rate limit reached, retry later."}),
    ]


def test_stream_hides_unexpected_error_details(client, rag):
    rag.generator = FailingGenerator(RuntimeError("secret internal state"))
    frames = parse_sse(client.post("/query/stream", json={"question": "q"}).text)
    assert frames[-1] == ("error", {"detail": "Internal server error."})
    assert "secret" not in json.dumps(frames)


def test_stream_validates_input(client):
    assert client.post("/query/stream", json={"question": ""}).status_code == 422
    assert client.post("/query/stream", json={"question": "q", "top_k": 0}).status_code == 422


def test_stream_is_documented_in_openapi(client):
    op = client.get("/openapi.json").json()["paths"]["/query/stream"]["post"]
    assert "text/event-stream" in op["responses"]["200"]["content"]
    assert "event" in op["description"].lower()


class UpFrontFailingGenerator:
    name = "up-front"

    def generate(self, question, hits):
        raise GenerationError("Claude rejected the credentials.")

    def stream(self, question, hits):
        raise GenerationError("Claude rejected the credentials.")
        yield  # makes this a lazy generator, like ClaudeGenerator.stream


def test_stream_returns_502_for_errors_before_the_first_event(client, rag):
    rag.generator = UpFrontFailingGenerator()
    r = client.post("/query/stream", json={"question": "q"})
    assert r.status_code == 502
    assert r.json()["detail"] == "Claude rejected the credentials."
    assert client.post("/query", json={"question": "q"}).status_code == 502


class NoDoneGenerator:
    name = "no-done"

    def generate(self, question, hits):
        raise NotImplementedError

    def __init__(self, events):
        self.events = events

    def stream(self, question, hits):
        yield from self.events


@pytest.mark.parametrize("events", [[], [TextDelta(text="Half an answer")]])
def test_stream_always_ends_with_done_or_error(client, rag, events):
    rag.generator = NoDoneGenerator(events)
    frames = parse_sse(client.post("/query/stream", json={"question": "q"}).text)
    assert frames[-1] == ("error", {"detail": "The answer stream ended unexpectedly."})
    assert [k for k, _ in frames[:-1]] == ["delta"] * len(events)
