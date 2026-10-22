from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from citerag import __version__
from citerag.config import Settings
from citerag.llm import GenerationError
from citerag.loaders import UnsupportedFileError
from citerag.models import Answer, DocumentInfo
from citerag.pipeline import RAG


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

    return app


app = create_app()
