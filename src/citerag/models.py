from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    doc_id: str
    source: str
    page: int | None = None
    text: str

    @property
    def label(self) -> str:
        return f"{self.source} (p. {self.page})" if self.page else self.source


class Hit(BaseModel):
    chunk: Chunk
    score: float


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    page: int | None = None
    quote: str = Field(description="The exact passage from the source that supports the answer.")


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    model: str


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    chunks: int


class TextDelta(BaseModel):
    """A piece of answer text, in generation order."""

    type: Literal["delta"] = "delta"
    text: str


class CitationEvent(BaseModel):
    """A source cited for the first time; `index` is the number used in `[n]` markers."""

    type: Literal["citation"] = "citation"
    index: int
    citation: Citation


class Done(BaseModel):
    """The complete answer. Always the last event of a stream."""

    type: Literal["done"] = "done"
    answer: Answer


StreamEvent = TextDelta | CitationEvent | Done
