import sys

import pytest
import typer
from conftest import BERLIN, LISBON
from typer.testing import CliRunner

from citerag.cli import app
from citerag.llm import GenerationError, events_from_answer
from citerag.models import Answer, Citation, CitationEvent, Done, TextDelta
from citerag.pipeline import RAG

runner = CliRunner()

CITATION = Citation(
    chunk_id="a:0", doc_id="a", source="lisbon.pdf", page=2, quote="It hosts\n  WS."
)


@pytest.fixture
def use_rag(rag, monkeypatch):
    rag.ingest("lisbon.txt", LISBON.encode())
    rag.ingest("berlin.md", BERLIN.encode())
    monkeypatch.setattr(RAG, "from_settings", lambda: rag)
    return rag


class ScriptedGenerator:
    """Streams a fixed list of events; `generate` returns the final answer."""

    name = "scripted"

    def __init__(self, events, error=None):
        self.events, self.error = events, error

    def generate(self, question, hits):
        if self.error:
            raise self.error
        return next(e.answer for e in self.events if isinstance(e, Done))

    def stream(self, question, hits):
        yield from self.events
        if self.error:
            raise self.error


def _ask(*args):
    return runner.invoke(app, ["ask", "Which city hosts Web Summit?", *args])


def test_ask_output_is_the_same_with_and_without_streaming(use_rag):
    streamed, plain = _ask(), _ask("--no-stream")
    assert streamed.exit_code == plain.exit_code == 0
    assert streamed.stdout == plain.stdout
    assert "Sources:\n  [1] lisbon.txt" in streamed.stdout


@pytest.mark.parametrize(
    "deltas",
    [
        ["\n", "  Lisbon ", "hosts", " WS.", " [1]", "\n\n"],
        ["", "Lisbon hosts WS. [1]"],
        ["   "],
    ],
)
def test_streamed_text_is_trimmed_like_the_final_answer(use_rag, deltas):
    text = "".join(deltas).strip()
    answer = Answer(answer=text, citations=[CITATION] if text else [], model="m")
    events = [TextDelta(text=d) for d in deltas] + [Done(answer=answer)]
    use_rag.generator = ScriptedGenerator(events)

    streamed, plain = _ask(), _ask("--no-stream")
    assert streamed.stdout == plain.stdout
    assert streamed.stdout.startswith(text + "\n")
    if text:
        assert '[1] lisbon.pdf, p. 2: "It hosts WS."' in streamed.stdout


def test_ask_flushes_each_delta_before_the_next_arrives(use_rag):
    seen = []

    class Recording(ScriptedGenerator):
        def stream(self, question, hits):
            for event in self.events:
                yield event
                seen.append(sys.stdout.buffer.getvalue().decode())

    answer = Answer(answer="Lisbon hosts WS.", citations=[], model="m")
    deltas = [TextDelta(text=t) for t in ["Lisbon ", "hosts ", "WS."]]
    use_rag.generator = Recording([*deltas, Done(answer=answer)])

    assert _ask().exit_code == 0
    assert seen[:3] == ["Lisbon", "Lisbon hosts", "Lisbon hosts WS."]


def test_ask_reports_mid_stream_errors_in_red_and_exits_1(use_rag):
    use_rag.generator = ScriptedGenerator(
        [TextDelta(text="Partial answer")], error=GenerationError("Claude rate limit reached.")
    )
    result = runner.invoke(app, ["ask", "q"], color=True)
    assert result.exit_code == 1
    assert result.stdout == "Partial answer\n"
    assert "Claude rate limit reached." in result.stderr
    assert "\x1b[31m" in result.stderr  # red


def test_ask_fails_if_the_stream_ends_without_an_answer(use_rag):
    use_rag.generator = ScriptedGenerator([TextDelta(text="Half")])
    result = _ask()
    assert result.exit_code == 1
    assert "ended unexpectedly" in result.stderr
    assert "Sources" not in result.stdout


def test_ask_no_stream_still_reports_errors(use_rag):
    use_rag.generator = ScriptedGenerator([], error=GenerationError("Bad credentials."))
    result = _ask("--no-stream")
    assert result.exit_code == 1
    assert "Bad credentials." in result.stderr
    assert result.stdout == ""


def test_ask_streaming_prints_sources_once(use_rag):
    answer = Answer(answer="WS. [1]", citations=[CITATION], model="m")
    events = list(events_from_answer(answer))
    assert any(isinstance(e, CitationEvent) for e in events)
    use_rag.generator = ScriptedGenerator(events)
    assert _ask().stdout == 'WS. [1]\n\nSources:\n  [1] lisbon.pdf, p. 2: "It hosts WS."\n'


def test_ask_has_a_documented_no_stream_flag():
    # Inspect the click command: rendered help is coloured by Rich on CI, splitting the text.
    ask = typer.main.get_command(app).commands["ask"]
    option = next(p for p in ask.params if p.name == "stream")
    assert option.secondary_opts == ["--no-stream"] and option.default is True
    assert option.help


@pytest.mark.parametrize("stream", ["--stream", "--no-stream"])
def test_ask_strips_terminal_escape_sequences(use_rag, stream):
    evil = "Hi \x1b[2J\x1b]8;;http://x\x07there\x9b31m\r\nok\tdone \x07"
    quote = Citation(chunk_id="a:0", doc_id="a", source="e\x1bvil.txt", quote="q\x1b[31m")
    answer = Answer(answer=evil, citations=[quote], model="m")
    use_rag.generator = ScriptedGenerator([TextDelta(text=evil), Done(answer=answer)])
    out = _ask(stream).stdout
    assert out == 'Hi [2J]8;;http://xthere31m\nok\tdone\n\nSources:\n  [1] evil.txt: "q[31m"\n'


def test_ask_rejects_non_positive_top_k(use_rag):
    result = runner.invoke(app, ["ask", "Lisbon?", "--top-k", "0"])
    assert result.exit_code != 0


def test_ask_accepts_per_query_mode_and_rejects_invalid(use_rag):
    calls = []
    original_ask = use_rag.ask

    def recording_ask(question, k=None, doc_ids=None, mode=None):
        calls.append(mode)
        return original_ask(question, k, doc_ids, mode)

    use_rag.ask = recording_ask
    result = runner.invoke(
        app, ["ask", "Which city hosts Web Summit?", "--mode", "keyword", "--no-stream"]
    )
    assert result.exit_code == 0
    assert calls == ["keyword"]
    assert use_rag.retrieval_mode == "hybrid"

    invalid = runner.invoke(app, ["ask", "question", "--mode", "semantic"])
    assert invalid.exit_code == 2


def test_ask_streaming_forwards_per_query_mode(use_rag):
    calls = []
    original_ask_stream = use_rag.ask_stream

    def recording_ask_stream(question, k=None, doc_ids=None, mode=None):
        calls.append(mode)
        return original_ask_stream(question, k, doc_ids, mode)

    use_rag.ask_stream = recording_ask_stream
    result = _ask("--mode", "keyword")
    assert result.exit_code == 0
    assert calls == ["keyword"]
    assert use_rag.retrieval_mode == "hybrid"
