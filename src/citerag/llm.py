"""Answer generators: Claude with native citations, and an offline extractive fallback."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol

import anthropic

from citerag.models import Answer, Citation, CitationEvent, Done, Hit, StreamEvent, TextDelta

NO_CONTEXT_ANSWER = "I couldn't find anything in the indexed documents that answers this question."

SYSTEM_PROMPT = """You answer questions using only the documents provided in the user turn.
Each document is a passage retrieved from the user's own files.

- Base every factual statement on those passages and cite them.
- If the passages don't contain the answer, say so plainly instead of guessing.
- Answer in the same language as the question."""


class GenerationError(RuntimeError):
    pass


class Generator(Protocol):
    name: str

    def generate(self, question: str, hits: list[Hit]) -> Answer: ...

    def stream(self, question: str, hits: list[Hit]) -> Iterator[StreamEvent]:
        """Yield the answer as events, ending with `Done`. Generators that can't stream
        natively can return `events_from_answer(self.generate(question, hits))`."""
        ...


def _citation(hit: Hit, quote: str) -> Citation:
    c = hit.chunk
    return Citation(chunk_id=c.id, doc_id=c.doc_id, source=c.source, page=c.page, quote=quote)


def events_from_answer(answer: Answer) -> Iterator[StreamEvent]:
    """Replay a finished answer as stream events, for generators that can't stream natively."""
    if answer.answer:
        yield TextDelta(text=answer.answer)
    for i, citation in enumerate(answer.citations, start=1):
        yield CitationEvent(index=i, citation=citation)
    yield Done(answer=answer)


class _CitationIndex:
    """Numbers cited passages in order of first appearance, so `[n]` markers stay stable."""

    def __init__(self, hits: list[Hit]) -> None:
        self._hits = hits
        self._seen: dict[tuple[int, str], int] = {}
        self.citations: list[Citation] = []

    def add(self, cite: Any) -> tuple[int, Citation | None] | None:
        """Return the marker number for an API citation, plus the Citation if it is new.

        Returns None for citations that don't point at one of the retrieved hits.
        """
        index = getattr(cite, "document_index", None)
        if index is None or not 0 <= index < len(self._hits):
            return None
        key = (index, cite.cited_text)
        if key in self._seen:
            return self._seen[key], None
        citation = _citation(self._hits[index], cite.cited_text.strip())
        self.citations.append(citation)
        self._seen[key] = len(self.citations)
        return self._seen[key], citation


class ClaudeGenerator:
    """Uses Claude's built-in citations: each retrieved chunk is sent as a document block,
    and the API returns the exact passages that back each part of the answer."""

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        max_tokens: int = 16000,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.name = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._client = client or anthropic.Anthropic()

    def _request(self, question: str, hits: list[Hit]) -> dict[str, Any]:
        documents = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": hit.chunk.text},
                "title": hit.chunk.label,
                "citations": {"enabled": True},
            }
            for hit in hits
        ]
        return {
            "model": self.name,
            "max_tokens": self._max_tokens,
            "system": SYSTEM_PROMPT,
            "output_config": {"effort": self._effort},
            # On a safety decline, the API re-runs the request on a fallback model it picks.
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
            "messages": [
                {"role": "user", "content": [*documents, {"type": "text", "text": question}]}
            ],
        }

    @contextmanager
    def _api_errors(self) -> Iterator[None]:
        try:
            yield
        except anthropic.AuthenticationError as exc:
            raise GenerationError(
                "Claude rejected the credentials. Set ANTHROPIC_API_KEY, "
                "or run with CITERAG_GENERATOR=extractive."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise GenerationError("Claude rate limit reached, retry later.") from exc
        except anthropic.APIStatusError as exc:
            raise GenerationError(f"Claude API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise GenerationError("Could not reach the Claude API.") from exc

    def generate(self, question: str, hits: list[Hit]) -> Answer:
        if not hits:
            return Answer(answer=NO_CONTEXT_ANSWER, citations=[], model=self.name)

        with self._api_errors():
            response = self._client.beta.messages.create(**self._request(question, hits))

        if response.stop_reason == "refusal":
            raise GenerationError("The model declined to answer this question.")

        parts: list[str] = []
        index = _CitationIndex(hits)
        for block in response.content:
            if block.type != "text":
                continue
            parts.append(block.text)
            for cite in block.citations or []:
                if marker := index.add(cite):
                    parts.append(f" [{marker[0]}]")

        return Answer(
            answer="".join(parts).strip(), citations=index.citations, model=response.model
        )

    def stream(self, question: str, hits: list[Hit]) -> Iterator[StreamEvent]:
        """Stream the answer as it is generated.

        Citation markers are emitted when their text block ends, using the same numbering and
        format as `generate()`. `Done.answer` is the whitespace-trimmed concatenation of the
        deltas. If the model declines mid-answer and the server falls back to another model,
        the partial text is kept and the fallback continues it (a non-streaming request would
        return only the fallback's text).
        """
        if not hits:
            yield from events_from_answer(
                Answer(answer=NO_CONTEXT_ANSWER, citations=[], model=self.name)
            )
            return

        parts: list[str] = []
        index = _CitationIndex(hits)
        pending: list[Any] = []  # citations of the text block currently streaming
        with (
            self._api_errors(),
            self._client.beta.messages.stream(**self._request(question, hits)) as stream,
        ):
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        parts.append(event.delta.text)
                        yield TextDelta(text=event.delta.text)
                    elif event.delta.type == "citations_delta":
                        pending.append(event.delta.citation)
                elif event.type == "content_block_stop":
                    for cite in pending:
                        if not (marker := index.add(cite)):
                            continue
                        number, new = marker
                        if new:
                            yield CitationEvent(index=number, citation=new)
                        parts.append(f" [{number}]")
                        yield TextDelta(text=f" [{number}]")
                    pending = []
            final = stream.get_final_message()

        if final.stop_reason == "refusal":
            raise GenerationError("The model declined to answer this question.")
        yield Done(
            answer=Answer(
                answer="".join(parts).strip(), citations=index.citations, model=final.model
            )
        )


class ExtractiveGenerator:
    """No LLM: returns the best-matching passages verbatim. Works offline and costs nothing."""

    name = "extractive"

    def __init__(self, max_passages: int = 3) -> None:
        self._max = max_passages

    def generate(self, question: str, hits: list[Hit]) -> Answer:
        if not hits:
            return Answer(answer=NO_CONTEXT_ANSWER, citations=[], model=self.name)
        top = hits[: self._max]
        answer = "\n\n".join(f"{hit.chunk.text} [{i}]" for i, hit in enumerate(top, start=1))
        return Answer(
            answer=answer, citations=[_citation(h, h.chunk.text) for h in top], model=self.name
        )

    def stream(self, question: str, hits: list[Hit]) -> Iterator[StreamEvent]:
        answer = self.generate(question, hits)
        if not hits:
            yield from events_from_answer(answer)
            return
        for i, citation in enumerate(answer.citations, start=1):
            yield TextDelta(text=("\n\n" if i > 1 else "") + citation.quote)
            yield CitationEvent(index=i, citation=citation)
            yield TextDelta(text=f" [{i}]")
        yield Done(answer=answer)
