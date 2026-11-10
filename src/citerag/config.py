from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from citerag.models import RetrievalMode
from citerag.rerank import DEFAULT_RERANK_MODEL


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables prefixed with CITERAG_."""

    model_config = SettingsConfigDict(env_prefix="CITERAG_", env_file=".env", extra="ignore")

    # Answer generation
    generator: Literal["claude", "extractive"] = "claude"
    claude_model: str = "claude-opus-5-5"
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_answer_tokens: int = 16000

    # Embeddings
    embedder: Literal["fastembed", "hash"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Vector store
    store: Literal["memory", "qdrant"] = "memory"
    data_dir: Path = Path("./data")
    qdrant_url: str | None = None
    qdrant_collection: str = "citerag"

    # Retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5
    retrieval_mode: RetrievalMode = "hybrid"
    rrf_k: int = Field(60, ge=0)
    reranker: Literal["none", "fastembed"] = "none"
    rerank_model: str = DEFAULT_RERANK_MODEL
    rerank_candidates: int = Field(20, ge=1)

    # API
    max_upload_mb: int = 25

    # URL ingestion. Private, loopback and link-local addresses are refused unless allowed.
    allow_private_urls: bool = False
    url_timeout: float = Field(10.0, gt=0)
