from types import SimpleNamespace

import pytest
from conftest import BERLIN, LISBON

from citerag.llm import NO_CONTEXT_ANSWER, ClaudeGenerator, ExtractiveGenerator, GenerationError
from citerag.models import Answer, Chunk, CitationEvent, Done, Hit, TextDelta


def _hits():
    return [
        Hit(chunk=Chunk(id="a:0", doc_id="a", source="lisbon.pdf", page=2, text=LISBON), score=0.9),
        Hit(chunk=Chunk(id="b:0", doc_id="b", source="berlin.txt", text=BERLIN), score=0.4),
    ]


def _cite(index, text):
    return SimpleNamespace(type="char_location", document_index=index, cited_text=text)


def _text_block(*deltas, citations=()):
    """Raw stream events for one text block: citations first, then text, like the API."""
    events = [SimpleNamespace(type="content_block_start")]
    events += [
        SimpleNamespace(
            type="content_block_delta", delta=SimpleNamespace(type="citations_delta", citation=c)
        )
        for c in citations
    ]
    events += [
        SimpleNamespace(
            type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=t)
        )
        for t in deltas
    ]
    events.append(SimpleNamespace(type="content_block_stop"))
    return events


class FakeStream:
    def __init__(self, events, final, error=None):
        self._events, self._final, self._error = events, final, error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        yield from self._events
        if self._error:
            raise self._error

    def get_final_message(self):
        return self._final


class FakeMessages:
    def __init__(self, stream=None, response=None):
        self._stream, self._response = stream, response
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return self._stream

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def _client(messages):
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def _final(stop_reason="end_turn", model="claude-opus-5-5"):
    return SimpleNamespace(stop_reason=stop_reason, model=model)


def test_claude_stream_emits_deltas_citations_and_done():
    first, second = _cite(0, "It hosts Web Summit."), _cite(1, "Berlin is the capital.")
    events = [
        *_text_block("Lisbon hosts ", "Web Summit.", citations=[first]),
        *_text_block(" Berlin is a capital.", citations=[second, first]),
        SimpleNamespace(type="message_stop"),
    ]
    messages = FakeMessages(stream=FakeStream(events, _final()))

    out = list(ClaudeGenerator(client=_client(messages)).stream("q", _hits()))

    text = "".join(e.text for e in out if isinstance(e, TextDelta))
    assert text == "Lisbon hosts Web Summit. [1] Berlin is a capital. [2] [1]"
    cites = [e for e in out if isinstance(e, CitationEvent)]
    assert [(c.index, c.citation.doc_id) for c in cites] == [(1, "a"), (2, "b")]
    assert cites[0].citation.page == 2

    done = out[-1]
    assert isinstance(done, Done)
    assert done.answer.answer == text
    assert [c.doc_id for c in done.answer.citations] == ["a", "b"]
    assert sum(isinstance(e, Done) for e in out) == 1

    sent = messages.calls[0]
    assert sent["fallbacks"] == "default" and sent["model"] == "claude-opus-5-5"
    assert all(b["citations"] == {"enabled": True} for b in sent["messages"][0]["content"][:-1])


def test_claude_stream_matches_generate_for_same_content():
    cite = _cite(0, "It hosts Web Summit.")
    events = [*_text_block("Lisbon hosts Web Summit.", citations=[cite]), *_text_block(" Done.")]
    response = SimpleNamespace(
        stop_reason="end_turn",
        model="claude-opus-5-5",
        content=[
            SimpleNamespace(type="text", text="Lisbon hosts Web Summit.", citations=[cite]),
            SimpleNamespace(type="text", text=" Done.", citations=None),
        ],
    )
    messages = FakeMessages(stream=FakeStream(events, _final()), response=response)
    gen = ClaudeGenerator(client=_client(messages))

    streamed = list(gen.stream("q", _hits()))[-1].answer
    assert streamed == gen.generate("q", _hits())
    assert messages.calls[0] == messages.calls[1]


def test_claude_stream_ignores_citations_to_unknown_documents():
    events = _text_block("Hi.", citations=[_cite(7, "x"), _cite(None, "y"), _cite(-1, "z")])
    messages = FakeMessages(stream=FakeStream(events, _final()))
    out = list(ClaudeGenerator(client=_client(messages)).stream("q", _hits()))
    assert out[-1].answer == Answer(answer="Hi.", citations=[], model="claude-opus-5-5")


def test_claude_stream_raises_on_refusal_after_partial_output():
    messages = FakeMessages(stream=FakeStream(_text_block("Partial"), _final("refusal")))
    stream = ClaudeGenerator(client=_client(messages)).stream("q", _hits())
    assert next(stream) == TextDelta(text="Partial")
    with pytest.raises(GenerationError, match="declined"):
        list(stream)


def test_claude_stream_wraps_api_errors_raised_mid_stream():
    import anthropic

    error = anthropic.APIConnectionError(request=None)
    messages = FakeMessages(stream=FakeStream(_text_block("Hi"), _final(), error=error))
    with pytest.raises(GenerationError, match="reach the Claude API"):
        list(ClaudeGenerator(client=_client(messages)).stream("q", _hits()))


def test_claude_stream_without_hits_skips_the_api():
    messages = FakeMessages()
    out = list(ClaudeGenerator(client=_client(messages)).stream("q", []))
    assert out[-1].answer.answer == NO_CONTEXT_ANSWER
    assert "".join(e.text for e in out if isinstance(e, TextDelta)) == NO_CONTEXT_ANSWER
    assert messages.calls == []


def test_extractive_stream_reassembles_generate():
    gen = ExtractiveGenerator()
    out = list(gen.stream("q", _hits()))
    expected = gen.generate("q", _hits())
    assert "".join(e.text for e in out if isinstance(e, TextDelta)) == expected.answer
    assert [e.index for e in out if isinstance(e, CitationEvent)] == [1, 2]
    assert out[-1] == Done(answer=expected)


def test_rag_ask_stream_ends_with_same_answer_as_ask(rag):
    rag.ingest("lisbon.txt", LISBON.encode())
    rag.ingest("berlin.md", BERLIN.encode())
    events = list(rag.ask_stream("Which city hosts Web Summit?"))
    assert isinstance(events[-1], Done)
    assert events[-1].answer == rag.ask("Which city hosts Web Summit?")


def test_rag_ask_stream_supports_generators_without_stream(rag):
    class LegacyGenerator:
        name = "legacy"

        def generate(self, question, hits):
            return Answer(answer="ok", citations=[], model=self.name)

    rag.generator = LegacyGenerator()
    events = list(rag.ask_stream("q"))
    assert events == [
        TextDelta(text="ok"),
        Done(answer=Answer(answer="ok", citations=[], model="legacy")),
    ]
