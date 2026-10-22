from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # API
    max_upload_mb: int = 25
