import json
import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from contextlib import asynccontextmanager
from functools import lru_cache
from itertools import chain

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from citerag import __version__
from citerag.config import Settings
from citerag.fetch import BlockedURLError, ResponseTooLargeError, URLFetchError
from citerag.jobs import Job, JobKind, JobManager, JobManagerClosedError, JobStatus, QueueFullError
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


JOB_RESPONSES = {
    202: {
        "model": Job,
        "description": "Accepted (`wait=false`): the document is ingested in the background. "
        "Poll the job at the `Location` header until its status is `succeeded` or `failed`.",
        "headers": {"Location": {"description": "URL of the job.", "schema": {"type": "string"}}},
    },
    503: _error("Too many ingestion jobs are pending, or the server is shutting down."),
}

WAIT_DESCRIPTION = (
    "Ingest before responding (201 with the document). With `false`, respond at once with "
    "202 and a job to poll at `GET /jobs/{id}`; errors found while ingesting are reported in "
    "the job instead of as an HTTP status."
)


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


_job_manager: JobManager | None = None
# `lru_cache` may build two instances when requests race on the first call, and a job
# submitted to the one that isn't kept could never be found again, so use a lock instead.
_job_manager_lock = threading.Lock()


def get_job_manager() -> JobManager:
    """The process-wide job manager, created on first use."""
    global _job_manager
    with _job_manager_lock:
        if _job_manager is None:
            _job_manager = JobManager.from_settings(get_settings())
        return _job_manager


def close_job_manager() -> None:
    """Shut down the job manager if one was created; the next use creates a new one."""
    global _job_manager
    with _job_manager_lock:
        manager, _job_manager = _job_manager, None
    if manager is not None:
        # Queued jobs are marked failed and the server stops at once; the interpreter still
        # lets jobs that are already running finish before the process exits.
        manager.shutdown(wait=False)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    close_job_manager()


def submit_job(
    request: Request, jobs: JobManager, kind: JobKind, source: str, fn: Callable[[], DocumentInfo]
) -> JSONResponse:
    """Queue an ingestion job and return the 202 response that points at it."""
    try:
        job = jobs.submit(kind, source, fn)
    except QueueFullError as exc:
        raise HTTPException(503, str(exc), headers={"Retry-After": "5"}) from exc
    except JobManagerClosedError as exc:
        raise HTTPException(503, "The server is shutting down.") from exc
    # A path rather than an absolute URL: behind a proxy, the scheme and host this process
    # sees may not be the ones the client used.
    root = request.scope.get("root_path", "").rstrip("/")
    location = root + request.app.url_path_for("get_job", job_id=job.id)
    return JSONResponse(
        job.model_dump(mode="json"), status_code=202, headers={"Location": location}
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="citerag",
        version=__version__,
        description="Ask questions about your documents and get answers with verifiable citations.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/jobs", response_model=list[Job])
    def list_jobs(
        status: JobStatus | None = None,
        job_manager: JobManager = Depends(get_job_manager),
    ) -> list[Job]:
        return job_manager.list(status)

    @app.get("/jobs/{job_id}", response_model=Job, responses={404: _error("Unknown job.")})
    def get_job(job_id: str, job_manager: JobManager = Depends(get_job_manager)) -> Job:
        """The status of an ingestion job. Finished jobs are kept for `CITERAG_JOB_TTL`
        seconds and only in the memory of the server process that accepted them."""
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found or expired.")
        return job

    @app.post("/documents", response_model=DocumentInfo, status_code=201, responses=JOB_RESPONSES)
    def upload(
        request: Request,
        file: UploadFile = File(...),
        wait: bool = Query(True, description=WAIT_DESCRIPTION),
        rag: RAG = Depends(get_rag),
        settings: Settings = Depends(get_settings),
        job_manager: JobManager = Depends(get_job_manager),
    ) -> DocumentInfo | JSONResponse:
        limit = settings.max_upload_mb * 1024 * 1024
        data = file.file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB.")
        filename = file.filename or "upload.txt"
        if not wait:
            return submit_job(
                request, job_manager, "file", filename, lambda: rag.ingest(filename, data)
            )
        try:
            return rag.ingest(filename, data)
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
