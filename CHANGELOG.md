# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-11-13

### Added

- DOCX loader using only the standard library: paragraphs, tables and text
  boxes, with size and element limits to protect memory (#24, #30).
- `.markdown`, `.rst`, `.csv` and `.json` files are ingested as plain text (#28, #31).
- Ingest documents from a URL. `citerag.fetch.Fetcher` downloads web pages and remote files with
  SSRF protection: the resolved IP is checked and pinned, redirects are re-checked, private,
  loopback, link-local and other internal ranges are refused, and size and total-time limits
  apply. Available as `RAG.ingest_url()` (#25, #35).
- `POST /documents/url` endpoint and `citerag ingest` accepting http(s) URLs alongside files and
  folders (#26, #37).
- `CITERAG_ALLOW_PRIVATE_URLS` (default `false`) and `CITERAG_URL_TIMEOUT` (default `10`
  seconds) settings for URL ingestion (#35).
- Per-query retrieval mode: `mode` field on `POST /query` and `POST /query/stream`, and
  `--mode` option on `citerag ask` (#16, #21).
- `citerag list` and `citerag delete DOC_ID` commands to show and remove indexed documents (#5, #23).
- `html2md-clean` added to the README's "Ecosystem" section (#38).

### Changed

- The HTML loader skips page boilerplate (navigation, headers, footers, sidebars) and keeps
  paragraph structure (#27, #33, #36).
- `httpx` is now a runtime dependency (used by the URL fetcher).

### Fixed

- The BM25 tokenizer keeps combining marks inside words and no longer drops kana voicing
  marks (#29, #32).
- Fetched HTML is decoded once, so pages whose headers and `<meta>` tag both name a charset no
  longer gain a stray byte-order mark (#36).

## [0.3.0] - 2026-11-06

### Added

- Okapi BM25 keyword index over the ingested chunks, with Unicode-aware tokenization
  (accent folding for Latin, Greek and Cyrillic; per-character tokens for CJK) (#13, #17).
- Hybrid retrieval: BM25 and vector rankings merged with Reciprocal Rank Fusion. Select it with
  `CITERAG_RETRIEVAL_MODE` (`hybrid`, `vector` or `keyword`) and tune it with `CITERAG_RRF_K` (#14, #18).
- `RAG.documents()` and `RAG.delete()`, which keep the vector store and the keyword index in
  sync (#18).
- Optional cross-encoder reranking after first-stage retrieval. Enable it with
  `CITERAG_RERANKER=fastembed` (needs the `local` extra), and set `CITERAG_RERANK_MODEL` and
  `CITERAG_RERANK_CANDIDATES` (#15, #20).
- `bm25-tiny` added to the README's "Ecosystem" section (#19).

### Changed

- The default retrieval mode is now `hybrid`. Set `CITERAG_RETRIEVAL_MODE=vector` to keep the
  v0.2.0 behaviour. Use `vector` when several replicas, or the CLI and the API, write to a shared
  Qdrant, because the keyword index lives in memory in each process.
- **Custom vector stores:** the `VectorStore` protocol now requires a `chunks()` method, which
  is used to rebuild the keyword index at start-up.

## [0.2.0] - 2026-11-03

### Added

- Streaming answers: `RAG.ask_stream()` yields `TextDelta`, `CitationEvent` and a final
  `Done` event carrying the complete `Answer` (#1, #6).
- `POST /query/stream` endpoint that sends the same events as Server-Sent Events (#2, #8).
- `citerag ask` streams the answer to the terminal by default; `--no-stream` restores the
  previous behaviour (#3, #9).
- Streaming section in the README with a `fetch`-based SSE reader and an animated terminal
  demo generated offline by `scripts/make_demo.py` (#4, #11).
- "Ecosystem" section in the README linking related projects (#10).

### Changed

- **Custom generators:** the `Generator` protocol now requires a `stream(question, hits)`
  method. Generators written against v0.1.0 must add it before upgrading.

## [0.1.0] - 2026-10-22

### Added

- First public release: FastAPI service and `citerag` CLI to ingest PDF, Markdown, HTML and text
  files and answer questions with verifiable citations, using Claude or an offline extractive
  generator, with an in-process vector store or Qdrant.

[Unreleased]: https://github.com/GuiDobelin/citerag/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/GuiDobelin/citerag/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/GuiDobelin/citerag/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/GuiDobelin/citerag/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/GuiDobelin/citerag/releases/tag/v0.1.0
