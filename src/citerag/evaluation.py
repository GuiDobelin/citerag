"""Measure retrieval quality on a labelled question set.

A dataset is JSON Lines: one object per line with a `question` and the passages that answer it.

    {"question": "Who hosts Web Summit?", "relevant": [{"source": "a.txt", "contains": "Summit"}]}
    {"question": "Revenue in 2025?", "relevant": [{"source": "report.pdf", "page": 3}], "id": "q2"}
    {"question": "Capital of Germany?", "relevant": ["berlin.md"]}

Relevant passages are described by their `source` (as `citerag list` shows it), optionally
narrowed by `page` and by a `contains` snippet, rather than by chunk id, so the judgements
still hold after changing the chunk size or re-ingesting. A plain string is shorthand for
`{"source": ...}`.

A `contains` snippet must fall inside one chunk to match, so keep it short (a phrase or a
sentence, well under `CITERAG_CHUNK_SIZE`).
"""

import json
import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, field_validator

from citerag.models import Chunk, Hit, RetrievalMode
from citerag.pipeline import RAG

_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().casefold()


class DatasetError(ValueError):
    """The evaluation dataset is malformed. The message names the offending line."""


class RelevantPassage(BaseModel):
    """A passage that answers a question: any chunk of `source` that also matches the optional
    `page` and `contains` filters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1)
    # Strict so a typo such as `"page": true` is an error rather than page 1.
    page: StrictInt | None = Field(default=None, ge=1)
    contains: str | None = None

    @field_validator("contains")
    @classmethod
    def _contains_has_text(cls, value: str | None) -> str | None:
        if value is not None and not _normalize(value):
            raise ValueError("contains must not be blank")
        return value

    def matches(self, chunk: Chunk) -> bool:
        """True if `chunk` is (part of) this passage. `contains` ignores case and whitespace."""
        if chunk.source != self.source:
            return False
        if self.page is not None and chunk.page != self.page:
            return False
        return self.contains is None or _normalize(self.contains) in _normalize(chunk.text)


class EvalCase(BaseModel):
    """One labelled question. `id` is optional and only used to identify it in reports."""

    model_config = ConfigDict(extra="forbid")

    question: str
    relevant: list[RelevantPassage] = Field(min_length=1)
    id: str | None = None

    @field_validator("question")
    @classmethod
    def _question_has_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @field_validator("relevant", mode="before")
    @classmethod
    def _expand_shorthand(cls, value: object) -> object:
        if isinstance(value, list):
            return [{"source": item} if isinstance(item, str) else item for item in value]
        return value

    @field_validator("relevant")
    @classmethod
    def _deduplicate(cls, value: list[RelevantPassage]) -> list[RelevantPassage]:
        # A passage listed twice would otherwise count twice in recall.
        return list(dict.fromkeys(value))


def parse_dataset(lines: Iterable[str]) -> list[EvalCase]:
    """Parse JSON Lines into cases. Blank lines are skipped; any other bad line raises
    `DatasetError` with its 1-based line number."""
    cases = []
    seen_ids: dict[str, int] = {}
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            case = EvalCase.model_validate(json.loads(line))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"line {number}: invalid JSON: {exc.msg}") from exc
        except RecursionError as exc:
            raise DatasetError(f"line {number}: JSON is nested too deeply") from exc
        except ValidationError as exc:
            raise DatasetError(f"line {number}: {_describe(exc)}") from exc
        if case.id is not None:
            if case.id in seen_ids:
                raise DatasetError(
                    f"line {number}: id {case.id!r} is already used on line {seen_ids[case.id]}"
                )
            seen_ids[case.id] = number
        cases.append(case)
    if not cases:
        raise DatasetError("the dataset has no questions")
    return cases


def load_dataset(path: str | Path) -> list[EvalCase]:
    """Read a JSON Lines dataset from disk (UTF-8, with or without a byte-order mark)."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{path} is not valid UTF-8") from exc
    # Split on newlines only: str.splitlines() would also break on U+2028 and friends, which
    # JSON allows unescaped inside strings. A trailing "\r" is JSON whitespace.
    return parse_dataset(text.split("\n"))


def _describe(exc: ValidationError) -> str:
    errors = exc.errors(include_url=False)
    if errors and errors[0]["type"] == "model_type" and not errors[0]["loc"]:
        return "each line must be a JSON object"
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'value'}: {error['msg']}"
        for error in errors
    )


def _check(relevant: Sequence[RelevantPassage], k: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1.")
    if not relevant:
        raise ValueError("relevant must not be empty.")


def _match_matrix(
    hits: Sequence[Hit], relevant: Sequence[RelevantPassage], k: int
) -> list[list[bool]]:
    """`matrix[i][j]` is whether the i-th of the top `k` hits matches the j-th distinct passage."""
    _check(relevant, k)
    passages = list(dict.fromkeys(relevant))
    return [[p.matches(hit.chunk) for p in passages] for hit in hits[:k]]


def _recall(matrix: list[list[bool]], passages: int) -> float:
    return sum(any(row[j] for row in matrix) for j in range(passages)) / passages


def _reciprocal_rank(matrix: list[list[bool]]) -> float:
    return next((1 / rank for rank, row in enumerate(matrix, start=1) if any(row)), 0.0)


def recall_at_k(hits: Sequence[Hit], relevant: Sequence[RelevantPassage], k: int) -> float:
    """Share of the relevant passages matched by at least one of the top `k` hits."""
    matrix = _match_matrix(hits, relevant, k)
    return _recall(matrix, len(set(relevant)))


def reciprocal_rank(hits: Sequence[Hit], relevant: Sequence[RelevantPassage], k: int) -> float:
    """1 / rank of the first of the top `k` hits that matches a relevant passage, else 0."""
    return _reciprocal_rank(_match_matrix(hits, relevant, k))


class RetrievedChunk(BaseModel):
    chunk_id: str
    source: str
    page: int | None = None
    relevant: bool


class CaseResult(BaseModel):
    id: str | None = None
    question: str
    recall: float
    reciprocal_rank: float
    hit: bool
    retrieved: list[RetrievedChunk]


class EvalReport(BaseModel):
    """Per-question results and their means. `mrr` is the mean reciprocal rank within the
    top `k`, and `hit_rate` the share of questions with at least one relevant hit."""

    k: int
    mode: RetrievalMode
    reranked: bool
    questions: int
    recall: float
    mrr: float
    hit_rate: float
    results: list[CaseResult]


def evaluate(
    rag: RAG, cases: Sequence[EvalCase], k: int | None = None, mode: RetrievalMode | None = None
) -> EvalReport:
    """Run every question through `rag.retrieve` and score the hits against the labels.

    Uses the pipeline as configured, so a reranker, if set, is part of what is measured.
    """
    if not cases:
        raise ValueError("cases must not be empty.")
    k = rag.top_k if k is None else k
    mode = rag.retrieval_mode if mode is None else mode
    results = []
    for case in cases:
        hits = rag.retrieve(case.question, k=k, mode=mode)
        matrix = _match_matrix(hits, case.relevant, k)
        rr = _reciprocal_rank(matrix)
        results.append(
            CaseResult(
                id=case.id,
                question=case.question,
                recall=_recall(matrix, len(case.relevant)),
                reciprocal_rank=rr,
                hit=rr > 0,
                retrieved=[
                    RetrievedChunk(
                        chunk_id=hit.chunk.id,
                        source=hit.chunk.source,
                        page=hit.chunk.page,
                        relevant=any(row),
                    )
                    for hit, row in zip(hits, matrix, strict=False)
                ],
            )
        )
    return EvalReport(
        k=k,
        mode=mode,
        reranked=rag.reranker is not None,
        questions=len(results),
        recall=math.fsum(r.recall for r in results) / len(results),
        mrr=math.fsum(r.reciprocal_rank for r in results) / len(results),
        hit_rate=sum(r.hit for r in results) / len(results),
        results=results,
    )
