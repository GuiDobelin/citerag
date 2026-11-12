import json
import logging
from collections.abc import Iterable, Iterator
from functools import lru_cache
from itertools import chain

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from citerag import __version__
from citerag.config import Settings
from citerag.fetch import BlockedURLError, ResponseTooLargeError, URLFetchError
from citerag.llm import GenerationError
from citerag.loaders import UnsupportedFileError
from citerag.models import Answer, DocumentInfo, Done, RetrievalMode, StreamEvent
from citerag.pipeline import RAG

logger = logging.getLogger(__name__)

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

STREAM_DESCRIPTION = """Same body as `POST /query`, but the answer is streamed as
[Server-Sent Events](https://html.spec.whatwg.org/multipage/server-sent-events.html):

- `delta`: `{"text": "..."}`, a piece of the answer, in order.
- `citation`: `{"index": 1, "citation": {...}}`, a source cited for the first time.
- `done`: the complete answer, same shape as the `POST /query` response. Always last on success.
- `error`: `{"detail": "..."}`, sent instead of `done` if generation fails mid-stream.

Errors before the first event return a normal HTTP error (502 for generation failures).
"""


def sse_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_events(events: Iterable[StreamEvent]) -> Iterator[str]:
    """Encode stream events as SSE frames, ending with `done` or `error`. Errors become an
    `error` frame, because the response status (200) has already been sent when they happen."""
    try:
        for event in events:
            if isinstance(event, Done):
                yield sse_frame("done", event.answer.model_dump(mode="json"))
                return
            yield sse_frame(event.type, event.model_dump(mode="json", exclude={"type"}))
        logger.error("Answer stream ended without a Done event")
        yield sse_frame("error", {"detail": "The answer stream ended unexpectedly."})
    except GenerationError as exc:
        yield sse_frame("error", {"detail": str(exc)})
    except Exception:
        logger.exception("Unexpected error while streaming an answer")
        yield sse_frame("error", {"detail": "Internal server error."})


class URLRequest(BaseModel):
    url: str = Field(
        min_length=1,
        max_length=2048,
        description="An http(s) URL of a web page or a PDF, DOCX, text or Markdown file.",
        examples=["https://example.com/guide"],
    )


class ErrorResponse(BaseModel):
    detail: str


def _error(description: str) -> dict:
    return {"model": ErrorResponse, "description": description}


URL_ERRORS = {
    400: _error("Invalid URL, or one that resolves to a private or internal address."),
    413: _error("The response is larger than `CITERAG_MAX_UPLOAD_MB`."),
    415: _error("The content type is not a supported document format."),
    # Listing 422 replaces FastAPI's default validation error response, so name both shapes.
    422: {
        "description": "Invalid request body, or a document with no extractable text.",
        "content": {
            "application/json": {
                "schema": {
                    "anyOf": [
                        {"$ref": "#/components/schemas/HTTPValidationError"},
                        {"$ref": "#/components/schemas/ErrorResponse"},
                    ]
                }
            }
        },
    },
    502: _error("The URL could not be fetched (network error, timeout or error status)."),
}


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    doc_ids: list[str] | None = Field(
        default=None, description="Restrict the search to these documents."
    )
    mode: RetrievalMode | None = Field(
        default=None,
        description="Retrieval mode for this query; defaults to the server setting.",
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

    @app.post(
        "/documents/url",
        response_model=DocumentInfo,
        status_code=201,
        responses=URL_ERRORS,
        summary="Ingest a URL",
    )
    def ingest_url(body: URLRequest, rag: RAG = Depends(get_rag)) -> DocumentInfo:
        """Download a web page or document and index it. The final URL, after redirects,
        becomes the document's `source`. Private and internal addresses are refused unless
        `CITERAG_ALLOW_PRIVATE_URLS=true`."""
        try:
            return rag.ingest_url(body.url)
        # The URL errors subclass ValueError, so they must be caught before the generic 422.
        except BlockedURLError as exc:
            raise HTTPException(400, str(exc)) from exc
        except ResponseTooLargeError as exc:
            raise HTTPException(413, str(exc)) from exc
        except URLFetchError as exc:
            logger.warning("Could not fetch %r: %s", body.url, exc)
            raise HTTPException(502, str(exc)) from exc
        except UnsupportedFileError as exc:
            raise HTTPException(415, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/documents", response_model=list[DocumentInfo])
    def list_documents(rag: RAG = Depends(get_rag)) -> list[DocumentInfo]:
        return rag.documents()

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str, rag: RAG = Depends(get_rag)) -> None:
        if not rag.delete(doc_id):
            raise HTTPException(404, "Document not found.")

    @app.post("/query", response_model=Answer)
    def query(body: QueryRequest, rag: RAG = Depends(get_rag)) -> Answer:
        try:
            return rag.ask(body.question, body.top_k, body.doc_ids, body.mode)
        except GenerationError as exc:
            raise HTTPException(502, str(exc)) from exc

    @app.post(
        "/query/stream",
        response_class=StreamingResponse,
        description=STREAM_DESCRIPTION,
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    def query_stream(body: QueryRequest, rag: RAG = Depends(get_rag)) -> StreamingResponse:
        # Retrieval and the first event run before the response starts, so errors that
        # happen up front (bad credentials, rate limits) still get a 502 like POST /query.
        events = rag.ask_stream(body.question, body.top_k, body.doc_ids, body.mode)
        try:
            first = [next(events)]
        except StopIteration:
            first = []
        except GenerationError as exc:
            raise HTTPException(502, str(exc)) from exc
        return StreamingResponse(
            sse_events(chain(first, events)), media_type="text/event-stream", headers=SSE_HEADERS
        )

    return app


app = create_app()
