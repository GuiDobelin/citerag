# citerag

**Ask questions about your documents and get answers with citations you can check.**

[![CI](https://github.com/GuiDobelin/citerag/actions/workflows/ci.yml/badge.svg)](https://github.com/GuiDobelin/citerag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

citerag is a self-hostable retrieval-augmented generation (RAG) service. Upload PDFs, Word (DOCX),
Markdown (`.md`, `.markdown`), reStructuredText (`.rst`), CSV, JSON, text or HTML files, then ask
questions over HTTP or from the command line. CSV and JSON are indexed as plain UTF-8 text. Every
answer points to the exact passage, file and page it came from, so you can verify it.

```text
$ citerag ask "What notice period does the contract require?"
Either party may terminate the agreement with 60 days' written notice. [1]

Sources:
  [1] services-agreement.pdf, p. 7: "Either party may terminate this Agreement upon sixty (60) days' written notice..."
```

<p align="center">
  <img src="https://raw.githubusercontent.com/GuiDobelin/citerag/main/docs/demo.svg" alt="citerag ingesting two Markdown files and streaming an answer with its sources" width="760">
</p>

<sub>The demo runs offline with the `extractive` generator; regenerate it with
`python scripts/make_demo.py`.</sub>

## Why citerag

- **Verifiable answers.** Uses Claude's built-in citations API: quotes are returned by the model
  against the retrieved passages, not pasted in afterwards.
- **Runs locally by default.** Embeddings are computed on your machine with
  [fastembed](https://github.com/qdrant/fastembed). Only the retrieved passages are sent to the LLM,
  and an offline `extractive` mode sends nothing at all.
- **Starts small, scales out.** Start with the built-in file-backed store; switch to
  [Qdrant](https://qdrant.tech) with one environment variable when you outgrow it.
- **Small and readable.** About 600 lines of typed Python, easy to fork and extend.

## Quickstart

### With Docker

```bash
git clone https://github.com/GuiDobelin/citerag && cd citerag
cp .env.example .env          # add your ANTHROPIC_API_KEY
docker compose up --build
```

The API is now at http://localhost:8000, with interactive docs at http://localhost:8000/docs.
Supported uploads are PDF, DOCX, Markdown, plain text and HTML. For Word files, the body text,
tables and text boxes are indexed; headers, footers, footnotes and comments are not (yet).

```bash
curl -F "file=@contract.pdf" http://localhost:8000/documents
curl -X POST http://localhost:8000/documents/url \
     -H "Content-Type: application/json" \
     -d '{"url": "https://docs.python.org/3/tutorial/venv.html"}'
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What notice period does the contract require?"}'
```

### With pip

```bash
pip install "citerag[local] @ git+https://github.com/GuiDobelin/citerag"
export ANTHROPIC_API_KEY=sk-ant-...

citerag ingest ./docs          # index a folder
citerag ingest https://docs.python.org/3/tutorial/venv.html   # or a web page
citerag ask "Summarise the refund policy"   # streams the answer; --no-stream waits for it
citerag list                   # show indexed documents and their ids
citerag delete DOC_ID          # remove a document
citerag eval qa.jsonl          # score retrieval on labelled questions (see Evaluation)
citerag serve                  # start the API on :8000
```

No API key? Run `CITERAG_GENERATOR=extractive citerag ask "..."` to get the best-matching
passages back verbatim, with no LLM involved.

## API

| Method   | Path                  | Description                                          |
| -------- | --------------------- | ---------------------------------------------------- |
| `POST`   | `/documents`          | Upload a file (multipart `file`). Returns `doc_id`.  |
| `POST`   | `/documents/url`      | `{"url": "https://..."}`: fetch and index a page.    |
| `GET`    | `/documents`          | List indexed documents.                              |
| `GET`    | `/jobs`               | List background ingestion jobs (`?status=failed`).   |
| `GET`    | `/jobs/{job_id}`      | Status and result of one background ingestion job.   |
| `DELETE` | `/documents/{doc_id}` | Remove a document and its chunks.                    |
| `POST`   | `/query`              | `{"question": "...", "top_k": 5, "doc_ids": [...]}`  |
| `POST`   | `/query/stream`       | Same body as `/query`; streams Server-Sent Events.   |
| `GET`    | `/health`             | Liveness check.                                      |

Both query endpoints accept an optional `mode` field: `vector`, `keyword`, or `hybrid`.
When omitted, the configured `CITERAG_RETRIEVAL_MODE` is used. The CLI supports the same
per-question override with `citerag ask "your question" --mode keyword`.

Example `/query` response:

```json
{
  "answer": "Either party may terminate the agreement with 60 days' written notice. [1]",
  "citations": [
    {
      "chunk_id": "9f2c41d07a3be2c1:14",
      "doc_id": "9f2c41d07a3be2c1",
      "source": "services-agreement.pdf",
      "page": 7,
      "quote": "Either party may terminate this Agreement upon sixty (60) days' written notice"
    }
  ],
  "model": "claude-opus-5-5"
}
```

### Streaming

`POST /query/stream` sends the answer as it is generated, as
[Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events):

```bash
curl -N -X POST http://localhost:8000/query/stream \
     -H "Content-Type: application/json" \
     -d '{"question": "What notice period does the contract require?"}'
```

```text
event: delta
data: {"text": "Either party may terminate the agreement with 60 days' written notice."}

event: citation
data: {"index": 1, "citation": {"chunk_id": "9f2c41d07a3be2c1:14", "source": "services-agreement.pdf", "page": 7, ...}}

event: delta
data: {"text": " [1]"}

event: done
data: {"answer": "...", "citations": [...], "model": "claude-opus-5-5"}
```

`done` has the same shape as the `/query` response and is always the last event. If generation
fails before the first event, the endpoint returns `502` like `/query`; if it fails after the
stream has started, an `event: error` with `{"detail": "..."}` is sent instead of `done`.

`EventSource` only supports `GET`, so in the browser read the stream with `fetch`:

```js
const res = await fetch("/query/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question: "What notice period does the contract require?" }),
});
if (!res.ok) throw new Error((await res.json()).detail); // e.g. 502 before the stream starts
const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
let buffer = "";
for (;;) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += value;
  const frames = buffer.split("\n\n");
  buffer = frames.pop(); // keep a partial frame for the next chunk
  for (const frame of frames) {
    const event = frame.match(/^event: (.*)$/m)?.[1];
    const data = JSON.parse(frame.match(/^data: (.*)$/m)?.[1] ?? "null");
    if (event === "delta") output.textContent += data.text;
    if (event === "done") showSources(data.citations);
    if (event === "error") showError(data.detail);
  }
}
```

From the command line, `citerag ask` streams by default; pass `--no-stream` to print the answer
only once it is complete.

### Background ingestion

Large files and slow web pages can take a while to index. Pass `wait=false` and the API
answers `202 Accepted` at once, with a job to poll in the `Location` header:

```bash
curl -i -X POST "http://localhost:8000/documents?wait=false" -F "file=@services-agreement.pdf"
```

```text
HTTP/1.1 202 Accepted
date: Thu, 19 Nov 2026 12:14:19 GMT
server: uvicorn
location: /jobs/b5cb2a357dbe4865b6d3519415647675
content-length: 214
content-type: application/json

{"id":"b5cb2a357dbe4865b6d3519415647675","kind":"file","source":"services-agreement.pdf","status":"queued","created_at":"2026-11-19T12:14:19.700695Z","started_at":null,"finished_at":null,"result":null,"error":null}
```

For a URL, send `"wait": false` in the body (or `?wait=false`, which takes precedence):

```bash
curl -X POST http://localhost:8000/documents/url \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com/handbook", "wait": false}'
```

Then poll the job until its `status` is `succeeded` or `failed`:

```bash
curl http://localhost:8000/jobs/b5cb2a357dbe4865b6d3519415647675
```

```json
{
  "id": "b5cb2a357dbe4865b6d3519415647675",
  "kind": "file",
  "source": "services-agreement.pdf",
  "status": "succeeded",
  "created_at": "2026-11-19T12:14:19.700695Z",
  "started_at": "2026-11-19T12:14:19.700911Z",
  "finished_at": "2026-11-19T12:14:21.204004Z",
  "result": {"doc_id": "9f2c41d07a3be2c1", "source": "services-agreement.pdf", "chunks": 38},
  "error": null
}
```

A job goes `queued` → `running` → `succeeded` or `failed`. On success, `result` is exactly what
the synchronous call returns (`doc_id`, `source`, `chunks`). On failure, `result` is `null` and
`error` holds the message the synchronous call would have sent as its error `detail`, for
example `"No extractable text found in 'scan.pdf'."` or `"'intranet.local' resolves to a
non-public address (10.0.0.5). Set CITERAG_ALLOW_PRIVATE_URLS=true to allow internal hosts."`.
Unexpected errors are logged on the server and reported only as `"Internal error while
ingesting the document."`.

Problems that can be found without doing the work are still reported at once: a file over
`CITERAG_MAX_UPLOAD_MB` gets `413`, and an invalid URL, a non-http(s) scheme or a private IP
address written literally gets `400`. Whether a host *name* resolves to a private address is
only known when the job fetches it, so that shows up as a failed job. When
`CITERAG_MAX_PENDING_JOBS` jobs are already queued or running, or the server is shutting down,
the API answers `503` (with `Retry-After: 5` when the queue is full); jobs still queued at
shutdown are marked `failed`.

`GET /jobs` lists the jobs this server process still remembers, newest first; filter with
`?status=queued|running|succeeded|failed`. Finished jobs are kept for `CITERAG_JOB_TTL` seconds
(or until `CITERAG_MAX_JOBS` are retained, oldest first), after which `GET /jobs/{job_id}`
returns `404`.

### Ingesting web pages

`POST /documents/url` and `citerag ingest` accept `http(s)://` URLs of web pages and of PDF,
DOCX, text or Markdown files; files, folders and URLs can be mixed in one `ingest` call. HTML
pages are reduced to their main content (navigation, footers and scripts are dropped), and the
final URL after redirects becomes the document's source in citations. The API answers 400 for
a blocked or invalid URL, 413 if the response exceeds `CITERAG_MAX_UPLOAD_MB`, 415 for an
unsupported content type, 422 if there is no text and 502 if the download fails. The CLI prints
those errors as `skipped` and moves on, exiting with status 1 only if nothing was indexed.

Fetching URLs on behalf of whoever can reach the API is a
[server-side request forgery][ssrf] risk, so by default citerag refuses any URL whose host
resolves to a private, loopback, link-local or otherwise non-public address (such as
`localhost`, `10.0.0.5` or the cloud metadata endpoint `169.254.169.254`). It checks every
redirect the same way and connects to the exact IP address it checked, so a DNS answer that
changes between the check and the request can't get around it. Proxy environment variables
are ignored for the same reason. To index pages on an internal network, for example a wiki
behind your VPN, set `CITERAG_ALLOW_PRIVATE_URLS=true`, and only on deployments where every
client is trusted.

[ssrf]: https://owasp.org/www-community/attacks/Server_Side_Request_Forgery

## Configuration

All settings are environment variables (or entries in `.env`).

| Variable                   | Default                  | Notes                                            |
| -------------------------- | ------------------------ | ------------------------------------------------ |
| `ANTHROPIC_API_KEY`        |                          | Required for the `claude` generator.             |
| `CITERAG_GENERATOR`        | `claude`                 | `claude` or `extractive` (offline, no LLM).      |
| `CITERAG_CLAUDE_MODEL`     | `claude-opus-5-5`        | Any Claude model ID.                             |
| `CITERAG_CLAUDE_EFFORT`    | `medium`                 | `low` … `max`. Lower is faster and cheaper.      |
| `CITERAG_EMBEDDER`         | `fastembed`              | `fastembed` (local) or `hash` (tests only).      |
| `CITERAG_EMBEDDING_MODEL`  | `BAAI/bge-small-en-v1.5` | Any [fastembed model][models]. For non-English documents, pick a multilingual one. |
| `CITERAG_STORE`            | `memory`                 | `memory` (file-backed) or `qdrant`.              |
| `CITERAG_DATA_DIR`         | `./data`                 | Where the `memory` store persists.               |
| `CITERAG_QDRANT_URL`       |                          | e.g. `http://qdrant:6333`.                       |
| `CITERAG_CHUNK_SIZE`       | `1000`                   | Characters per chunk.                            |
| `CITERAG_CHUNK_OVERLAP`    | `150`                    | Overlap between chunks.                          |
| `CITERAG_TOP_K`            | `5`                      | Chunks retrieved per question.                   |
| `CITERAG_RETRIEVAL_MODE`   | `hybrid`                 | `hybrid` (BM25 + vectors), `vector` or `keyword`. |
| `CITERAG_RRF_K`            | `60`                     | Reciprocal Rank Fusion constant for `hybrid`.    |
| `CITERAG_RERANKER`         | `none`                   | `none` or `fastembed` (local cross-encoder).     |
| `CITERAG_RERANK_MODEL`     | `Xenova/ms-marco-MiniLM-L-6-v2` | Any [fastembed reranker][rerankers]. Use `jinaai/jina-reranker-v2-base-multilingual` for non-English documents. |
| `CITERAG_RERANK_CANDIDATES` | `20`                    | Chunks the reranker rescores (at least `top_k`). |
| `CITERAG_MAX_UPLOAD_MB`    | `25`                     | Size limit for uploads and downloaded URLs.      |
| `CITERAG_ALLOW_PRIVATE_URLS` | `false`                | Allow URLs on private or internal addresses.     |
| `CITERAG_URL_TIMEOUT`      | `10`                     | Seconds allowed to download a URL, redirects included. |
| `CITERAG_JOB_WORKERS`      | `2`                      | Background ingestion jobs that run at the same time. |
| `CITERAG_JOB_TTL`          | `3600`                   | Seconds a finished job stays visible at `/jobs`. |
| `CITERAG_MAX_JOBS`         | `1000`                   | Jobs retained in memory; the oldest finished ones are dropped first. |
| `CITERAG_MAX_PENDING_JOBS` | `20`                     | Queued plus running jobs before new ones get `503`. Each holds its upload in memory. |

[models]: https://qdrant.github.io/fastembed/examples/Supported_Models/
[rerankers]: https://qdrant.github.io/fastembed/examples/Supported_Models/#supported-rerank-cross-encoder-models

## How it works

```text
upload ─▶ load (PDF/DOCX/text/HTML) ─▶ chunk ─┬▶ embed (local) ─▶ vector store ─┐
                                              └▶ BM25 keyword index ────────────┤
                                                                                │
question ─▶ vector + keyword search ─▶ RRF ─▶ (rerank) ─▶ top-k chunks ◀────────┘
                                                            │
                                                            ▼
          Claude (each chunk sent as a citable document) ─▶ answer + quotes
```

Retrieval is **hybrid** by default. Vector search finds passages that mean the same thing as
the question; BM25 finds exact terms that embeddings tend to blur, such as error codes, names
or IDs. Each side returns `4 × top_k` candidates and the two rankings are merged with
[Reciprocal Rank Fusion][rrf] (`score = Σ 1 / (60 + rank)`), which only looks at positions, so
the two scores never need calibrating. The keyword index is kept in memory and rebuilt from the
store when the app starts.

[rrf]: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

Reranking is optional. With `CITERAG_RERANKER=fastembed`, a small cross-encoder reads the
question together with each of the best 20 candidates and reorders them before the top `k`
go to Claude. Embeddings compare the question and a passage that were encoded separately; a
cross-encoder sees both at once, so it is better at telling a passage that *answers* the
question from one that merely shares its words. It costs latency (about 0.3 s per question
for 20 full-size chunks on a laptop CPU with the default model), and the model (~90 MB) is
downloaded on first use.

Documents are identified by a hash of their content, so uploading the same file twice replaces
it instead of duplicating it.

## Scaling

- **More documents:** set `CITERAG_STORE=qdrant`. `docker compose --profile qdrant up` starts
  one alongside the API.
- **More traffic:** with Qdrant the vectors live outside the API, so you can run several
  replicas behind a load balancer. The BM25 index is per process, though: a replica only sees
  keyword matches for documents it ingested itself or that existed when it built the index
  (on its first query). The same applies to a running API when you ingest into the same
  Qdrant with the CLI. Until the keyword index is shared, use `CITERAG_RETRIEVAL_MODE=vector`
  in those setups, or restart the API after ingesting.
- **Background jobs:** jobs live in the memory of the API process that accepted them. They
  are lost on restart, and with several replicas each one only knows its own jobs, so poll
  `GET /jobs/{job_id}` through sticky sessions or on the same instance. The indexed documents
  themselves are in the store and are not affected.
- **Costs:** lower `CITERAG_CLAUDE_EFFORT` or `CITERAG_TOP_K`, or use a smaller model.

## Evaluation

`citerag eval` measures how well retrieval finds the passages that answer your questions,
on your own corpus, without calling an LLM. Write a labelled dataset as JSON Lines, one
question per line:

```json
{"id": "summit", "question": "Which city hosts Web Summit?", "relevant": [{"source": "lisbon.txt", "contains": "Web Summit"}]}
{"question": "Revenue in 2025?", "relevant": [{"source": "report.pdf", "page": 3}]}
{"question": "Where are Python developers hired?", "relevant": ["berlin.md"]}
```

- `question` (required) is sent to retrieval as is.
- `relevant` (required, non-empty) lists the passages that answer it. `source` is the document
  name as `citerag list` shows it (the file name, or the final URL of a web page). `page`
  narrows it to one page of a PDF, and `contains` to the chunks that include a snippet
  (case and whitespace are ignored). Keep `contains` to a phrase: it must fit inside one
  chunk. A plain string such as `"berlin.md"` is shorthand for `{"source": "berlin.md"}`.
  Judging by source rather than chunk id keeps the dataset valid when you change the chunk
  size or re-ingest.
- `id` (optional, unique) names the question in reports.

Then score it against the index:

```bash
citerag eval qa.jsonl                    # configured mode and CITERAG_TOP_K
citerag eval qa.jsonl --k 2 --mode all --details
```

```text
warning: source 'madrid.txt' is not indexed.
Mode     Recall@2  MRR@2  nDCG@2  Hit rate@2
-------  --------  -----  ------  ----------
vector   0.750     0.750  0.750   0.750
keyword  0.750     0.750  0.750   0.750
hybrid   0.750     0.750  0.750   0.750

4 questions

Missed (vector):
  - madrid: retrieved [lisbon.txt, berlin.md]
...
```

All metrics look at the top `k` hits and are averaged over the questions:

| Metric | Meaning |
|---|---|
| Recall@k | Share of a question's relevant passages found by at least one hit |
| MRR@k | 1 / rank of the first relevant hit (0 if none) |
| nDCG@k | Rewards relevant hits near the top; binary relevance, each passage counted once |
| Hit rate@k | Share of questions with at least one relevant hit |

`--mode all` runs vector, keyword and hybrid retrieval side by side. If `CITERAG_RERANKER` is
set, the reranker is part of what is measured and the output says so. `--details` lists the
questions with no relevant hit and the sources retrieved instead. `--json` prints a JSON
array with one full report per mode (per-question scores and retrieved chunks included),
handy for tracking regressions in CI. A `source` that is not indexed only prints a warning
(it scores as a miss); a missing file or a malformed line exits with status 1 and names the
line.

## Roadmap

- [x] Background ingestion queue for large uploads
- [x] Hybrid search (BM25 + vectors with RRF)
- [x] Optional cross-encoder reranking
- [x] Streaming answers (SSE)
- [ ] API keys and per-tenant collections
- [ ] Evaluation harness for retrieval quality
- [x] DOCX loader
- [x] Ingest web pages by URL

Ideas and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Ecosystem

Small, standalone tools built alongside citerag:

- [llm-cost](https://github.com/GuiDobelin/llm-cost): estimate token counts and API cost per
  model from the CLI or Python, including prompt-cache and long-context pricing.
- [bm25-tiny](https://github.com/GuiDobelin/bm25-tiny): Okapi BM25 / BM25+ keyword search in
  pure Python with no dependencies, with incremental updates, JSON persistence and a CLI.
- [html2md-clean](https://github.com/GuiDobelin/html2md-clean): turn web pages into clean
  Markdown with only the standard library, dropping navigation and boilerplate while keeping
  headings, lists, links, code and tables.
- [pdf2chunks](https://github.com/GuiDobelin/pdf2chunks): turn PDFs into JSONL chunks with page
  numbers and metadata, removing running headers, footers and hyphenation before chunking.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

If you change the CLI output, regenerate the README demo with `python scripts/make_demo.py`;
a test fails while `docs/demo.svg` is out of date.

Tests use a fake LLM client and a hashing embedder, so they run offline and free.

## License

[MIT](LICENSE)
