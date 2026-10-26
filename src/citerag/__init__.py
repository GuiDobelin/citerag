"""citerag: answers about your documents, with citations you can check."""

from citerag.models import Answer, Chunk, Citation, CitationEvent, Done, StreamEvent, TextDelta
from citerag.pipeline import RAG

__all__ = [
    "RAG",
    "Answer",
    "Chunk",
    "Citation",
    "CitationEvent",
    "Done",
    "StreamEvent",
    "TextDelta",
]
__version__ = "0.1.0"
