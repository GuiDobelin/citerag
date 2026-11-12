from pathlib import Path

import httpx
import typer

from citerag.llm import GenerationError
from citerag.models import Answer, Citation, Done, RetrievalMode, TextDelta
from citerag.pipeline import RAG, RETRIEVAL_MODES

app = typer.Typer(
    help="Answers about your documents, with citations you can check.", no_args_is_help=True
)


def _is_url(arg: str) -> bool:
    return arg.strip().lower().startswith(("http://", "https://"))


@app.command()
def ingest(
    sources: list[str] = typer.Argument(..., help="Files, folders or http(s) URLs to index."),
):
    """Index files, folders and web pages into the local store.

    Exits with status 1 if nothing could be indexed."""
    # Check every local path before indexing anything, so a typo doesn't leave half a batch.
    missing = [s for s in sources if not _is_url(s) and not Path(s).exists()]
    if missing:
        raise typer.BadParameter(
            f"Path '{_one_line(missing[0])}' does not exist.", param_hint="SOURCES"
        )
    rag = RAG.from_settings()
    indexed = 0
    for source in sources:
        if _is_url(source):
            indexed += _ingest_one(rag, source.strip())
            continue
        path = Path(source)
        for f in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if f.is_file():
                indexed += _ingest_one(rag, f)
    if not indexed:
        typer.secho("Nothing was indexed.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


def _ingest_one(rag: RAG, source: str | Path) -> bool:
    """Index one file or URL and print the outcome. Returns whether it was indexed."""
    label = _one_line(str(source))
    try:
        if isinstance(source, Path):
            info = rag.ingest(source.name, source.read_bytes())
        else:
            info = rag.ingest_url(source)
    # ValueError covers unsupported files and URL fetch errors; OSError an unreadable file.
    except (ValueError, OSError) as exc:
        typer.secho(f"skipped  {label}: {_one_line(str(exc))}", fg=typer.colors.YELLOW)
        return False
    if isinstance(source, str) and info.source != _normalized(source):
        # A redirected URL is stored under its final address, which `list` will show.
        label += f" -> {_one_line(info.source)}"
    typer.echo(f"indexed  {label}  ({info.chunks} chunks, id={info.doc_id})")
    return True


def _normalized(url: str) -> str:
    """The URL as the fetcher reports it when there is no redirect."""
    try:
        return str(httpx.URL(url))
    except httpx.InvalidURL:
        return url


# C0/C1 control characters except tab and newline, so text from indexed documents can't send
# escape sequences to the user's terminal.
_CONTROL_CHARS = dict.fromkeys(
    c for c in [*range(0x20), *range(0x7F, 0xA0)] if c not in (0x09, 0x0A)
)


def _printable(text: str) -> str:
    return text.translate(_CONTROL_CHARS)


def _one_line(text: str) -> str:
    """`_printable`, with line breaks folded to spaces so a value prints on a single line."""
    return " ".join(_printable(text).splitlines())


class _StreamPrinter:
    """Echo answer deltas as they arrive, trimmed like `Answer.answer`: leading whitespace is
    dropped and trailing whitespace is held back until more text follows it."""

    def __init__(self) -> None:
        self._started = False
        self._held = ""

    def write(self, text: str) -> None:
        text = _printable(text)
        if not self._started:
            text = text.lstrip()
            if not text:
                return
            self._started = True
        body = text.rstrip()
        if body:
            typer.echo(self._held + body, nl=False)
            self._held = text[len(body) :]
        else:
            self._held += text

    def end(self) -> None:
        """Terminate a partial line, e.g. before an error message."""
        if self._started:
            typer.echo()


def _print_sources(citations: list[Citation]) -> None:
    if not citations:
        return
    typer.echo("\nSources:")
    for i, c in enumerate(citations, start=1):
        where = _printable(f"{c.source}, p. {c.page}" if c.page else c.source)
        quote = _printable(" ".join(c.quote.split()))[:160]
        typer.echo(f'  [{i}] {where}: "{quote}"')


def _ask_streaming(
    rag: RAG, question: str, top_k: int | None, mode: RetrievalMode | None
) -> Answer:
    printer = _StreamPrinter()
    try:
        for event in rag.ask_stream(question, top_k, mode=mode):
            if isinstance(event, TextDelta):
                printer.write(event.text)
            elif isinstance(event, Done):
                typer.echo()  # the answer's final newline, as with `--no-stream`
                return event.answer
        raise GenerationError("The answer stream ended unexpectedly.")
    except BaseException:
        printer.end()
        raise


@app.command()
def ask(
    question: str,
    top_k: int = typer.Option(None, min=1, help="Number of chunks to retrieve."),
    stream: bool = typer.Option(
        True, "--stream/--no-stream", help="Print the answer as it is generated."
    ),
    mode: str | None = typer.Option(
        None, "--mode", help="Retrieval mode: vector, keyword, or hybrid."
    ),
):
    """Ask a question about the indexed documents."""
    if mode is not None and mode not in RETRIEVAL_MODES:
        raise typer.BadParameter(f"must be one of {', '.join(RETRIEVAL_MODES)}")
    rag = RAG.from_settings()
    try:
        if stream:
            answer = _ask_streaming(rag, question, top_k, mode)
        else:
            answer = rag.ask(question, top_k, mode=mode)
            typer.echo(_printable(answer.answer).strip())
    except GenerationError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    _print_sources(answer.citations)


@app.command("list")
def list_documents():
    """List the indexed documents."""
    docs = RAG.from_settings().documents()
    if not docs:
        typer.echo("No documents indexed.")
        return
    for d in docs:
        typer.echo(f"{_one_line(d.doc_id)}  {_one_line(d.source)}  ({d.chunks} chunks)")


@app.command()
def delete(doc_id: str = typer.Argument(..., help="Id of the document, as shown by `list`.")):
    """Remove a document from the index."""
    if not RAG.from_settings().delete(doc_id):
        typer.secho(f"No document with id '{_one_line(doc_id)}'.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    typer.echo(f"deleted  {_one_line(doc_id)}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("citerag.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
