FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CITERAG_DATA_DIR=/data \
    HF_HOME=/models

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[all]" \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /data /models && chown app /data /models

USER app
# Download the default embedding model at build time so the container starts fast.
RUN python -c "from citerag.embeddings import FastEmbedEmbedder; FastEmbedEmbedder('BAAI/bge-small-en-v1.5')"

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "citerag.api:app", "--host", "0.0.0.0", "--port", "8000"]
