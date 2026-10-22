"""Answer generators: Claude with native citations, and an offline extractive fallback."""

from typing import Protocol

import anthropic

from citerag.models import Answer, Citation, Hit

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


def _citation(hit: Hit, quote: str) -> Citation:
    c = hit.chunk
    return Citation(chunk_id=c.id, doc_id=c.doc_id, source=c.source, page=c.page, quote=quote)


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

    def generate(self, question: str, hits: list[Hit]) -> Answer:
        if not hits:
            return Answer(answer=NO_CONTEXT_ANSWER, citations=[], model=self.name)

        documents = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": hit.chunk.text},
                "title": hit.chunk.label,
                "citations": {"enabled": True},
            }
            for hit in hits
        ]
        try:
            response = self._client.beta.messages.create(
                model=self.name,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                output_config={"effort": self._effort},
                # On a safety decline, the API re-runs the request on a fallback model it picks.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[
                    {"role": "user", "content": [*documents, {"type": "text", "text": question}]}
                ],
            )
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

        if response.stop_reason == "refusal":
            raise GenerationError("The model declined to answer this question.")

        parts: list[str] = []
        citations: list[Citation] = []
        seen: dict[tuple[int, str], int] = {}
        for block in response.content:
            if block.type != "text":
                continue
            parts.append(block.text)
            for cite in block.citations or []:
                index = getattr(cite, "document_index", None)
                if index is None or index >= len(hits):
                    continue
                key = (index, cite.cited_text)
                if key not in seen:
                    seen[key] = len(citations) + 1
                    citations.append(_citation(hits[index], cite.cited_text.strip()))
                parts.append(f" [{seen[key]}]")

        return Answer(answer="".join(parts).strip(), citations=citations, model=response.model)


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
