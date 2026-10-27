import json
import logging
from collections.abc import Iterator
from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from citerag import __version__
from citerag.config import Settings
from citerag.llm import GenerationError
from citerag.loaders import UnsupportedFileError
from citerag.models import Answer, CitationEvent, DocumentInfo, Done, StreamEvent, TextDelta
from citerag.pipeline import RAG

logger = logging.getLogger(__name__)

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

STREAM_DESCRIPTION = """Same body as `POST /query`, but the answer is streamed as
[Server-Sent Events](https://html.spec.whatwg.org/multipage/server-sent-events.html):

- `delta`: `{"text": "..."}`, a piece of the answer, in order.
- `citation`: `{"index": 1, "citation": {...}}`, a source cited for the first time.
- `done`: the complete answer, same shape as the `POST /query` response. Always last on success.
- `error`: `{"detail": "..."}`, sent instead of `done` if generation fails mid-stream.
"""


def sse_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_events(events: Iterator[StreamEvent]) -> Iterator[str]:
    """Encode stream events as SSE frames. Errors become an `error` frame, because the
    response status (200) has already been sent when they happen."""
    try:
        for event in events:
            if isinstance(event, TextDelta):
                yield sse_frame("delta", {"text": event.text})
            elif isinstance(event, CitationEvent):
                yield sse_frame(
                    "citation", {"index": event.index, "citation": event.citation.model_dump()}
                )
            elif isinstance(event, Done):
                yield sse_frame("done", event.answer.model_dump())
    except GenerationError as exc:
        yield sse_frame("error", {"detail": str(exc)})
    except Exception:
        logger.exception("Unexpected error while streaming an answer")
        yield sse_frame("error", {"detail": "Internal server error."})


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    doc_ids: list[str] | None = Field(
        default=None, description="Restrict the search to these documents."
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_rag() -> RAG:
    return RAG.from_settings(get_settings())


def create_app() -> FastAPI:
    app = FastAPI(
        title="citerag",
        version=__version__,
        description="Ask questions about your documents and get answers with verifiable citations.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.post("/documents", response_model=DocumentInfo, status_code=201)
    def upload(
        file: UploadFile = File(...),
        rag: RAG = Depends(get_rag),
        settings: Settings = Depends(get_settings),
    ) -> DocumentInfo:
        limit = settings.max_upload_mb * 1024 * 1024
        data = file.file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB.")
        try:
            return rag.ingest(file.filename or "upload.txt", data)
        except UnsupportedFileError as exc:
            raise HTTPException(415, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/documents", response_model=list[DocumentInfo])
    def list_documents(rag: RAG = Depends(get_rag)) -> list[DocumentInfo]:
        return rag.store.documents()

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str, rag: RAG = Depends(get_rag)) -> None:
        if not rag.store.delete(doc_id):
            raise HTTPException(404, "Document not found.")

    @app.post("/query", response_model=Answer)
    def query(body: QueryRequest, rag: RAG = Depends(get_rag)) -> Answer:
        try:
            return rag.ask(body.question, body.top_k, body.doc_ids)
        except GenerationError as exc:
            raise HTTPException(502, str(exc)) from exc

    @app.post(
        "/query/stream",
        response_class=StreamingResponse,
        description=STREAM_DESCRIPTION,
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    def query_stream(body: QueryRequest, rag: RAG = Depends(get_rag)) -> StreamingResponse:
        # Retrieval runs here, before the response starts, so its errors still get a status.
        events = rag.ask_stream(body.question, body.top_k, body.doc_ids)
        return StreamingResponse(
            sse_events(events), media_type="text/event-stream", headers=SSE_HEADERS
        )

    return app


app = create_app()
