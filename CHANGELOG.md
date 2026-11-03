# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/GuiDobelin/citerag/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/GuiDobelin/citerag/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/GuiDobelin/citerag/releases/tag/v0.1.0
