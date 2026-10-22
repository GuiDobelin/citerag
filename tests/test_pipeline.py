from types import SimpleNamespace

import pytest
from conftest import BERLIN, LISBON

from citerag.llm import NO_CONTEXT_ANSWER, ClaudeGenerator, GenerationError
from citerag.models import Chunk, Hit


def test_ingest_and_ask_returns_cited_answer(rag):
    rag.ingest("lisbon.txt", LISBON.encode())
    rag.ingest("berlin.md", BERLIN.encode())

    answer = rag.ask("Which city hosts Web Summit?")
    assert answer.citations[0].source == "lisbon.txt"
    assert "Web Summit" in answer.answer


def test_reingesting_same_file_does_not_duplicate(rag):
    first = rag.ingest("lisbon.txt", LISBON.encode())
    second = rag.ingest("lisbon.txt", LISBON.encode())
    assert first.doc_id == second.doc_id
    assert [d.chunks for d in rag.store.documents()] == [first.chunks]


def test_ask_with_empty_index(rag):
    assert rag.ask("anything").answer == NO_CONTEXT_ANSWER


def test_ingest_rejects_empty_file(rag):
    with pytest.raises(ValueError):
        rag.ingest("empty.txt", b"   ")


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _fake_client(response):
    messages = FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def _hits():
    return [
        Hit(chunk=Chunk(id="a:0", doc_id="a", source="lisbon.pdf", page=2, text=LISBON), score=0.9),
        Hit(chunk=Chunk(id="b:0", doc_id="b", source="berlin.txt", text=BERLIN), score=0.4),
    ]


def test_claude_generator_maps_citations_to_chunks():
    cite = SimpleNamespace(
        type="char_location", document_index=0, cited_text="It hosts Web Summit."
    )
    response = SimpleNamespace(
        stop_reason="end_turn",
        model="claude-opus-5-5",
        content=[
            SimpleNamespace(type="text", text="Lisbon hosts Web Summit.", citations=[cite]),
            SimpleNamespace(type="text", text=" It is in Portugal.", citations=[cite]),
        ],
    )
    client, messages = _fake_client(response)

    answer = ClaudeGenerator(client=client).generate("Who hosts Web Summit?", _hits())

    assert answer.answer == "Lisbon hosts Web Summit. [1] It is in Portugal. [1]"
    assert len(answer.citations) == 1
    assert answer.citations[0].source == "lisbon.pdf" and answer.citations[0].page == 2

    sent = messages.calls[0]
    docs = [b for b in sent["messages"][0]["content"] if b["type"] == "document"]
    assert len(docs) == 2 and docs[0]["citations"] == {"enabled": True}
    assert sent["model"] == "claude-opus-5-5"
    assert sent["fallbacks"] == "default"


def test_claude_generator_skips_call_without_hits():
    client, messages = _fake_client(None)
    assert ClaudeGenerator(client=client).generate("q", []).answer == NO_CONTEXT_ANSWER
    assert messages.calls == []


def test_claude_generator_raises_on_refusal():
    response = SimpleNamespace(stop_reason="refusal", model="claude-opus-5-5", content=[])
    client, _ = _fake_client(response)
    with pytest.raises(GenerationError):
        ClaudeGenerator(client=client).generate("q", _hits())
