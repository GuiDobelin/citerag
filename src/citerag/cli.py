from pathlib import Path

import typer

from citerag.llm import GenerationError
from citerag.models import Answer, Citation, Done, TextDelta
from citerag.pipeline import RAG

app = typer.Typer(
    help="Answers about your documents, with citations you can check.", no_args_is_help=True
)


@app.command()
def ingest(paths: list[Path] = typer.Argument(..., exists=True, help="Files or folders to index.")):
    """Index files into the local store."""
    rag = RAG.from_settings()
    files = [f for p in paths for f in (sorted(p.rglob("*")) if p.is_dir() else [p]) if f.is_file()]
    for f in files:
        try:
            info = rag.ingest(f.name, f.read_bytes())
            typer.echo(f"indexed  {f}  ({info.chunks} chunks, id={info.doc_id})")
        except ValueError as exc:
            typer.secho(f"skipped  {f}: {exc}", fg=typer.colors.YELLOW)


# C0/C1 control characters except tab and newline, so text from indexed documents can't send
# escape sequences to the user's terminal.
_CONTROL_CHARS = dict.fromkeys(
    c for c in [*range(0x20), *range(0x7F, 0xA0)] if c not in (0x09, 0x0A)
)


def _printable(text: str) -> str:
    return text.translate(_CONTROL_CHARS)


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


def _ask_streaming(rag: RAG, question: str, top_k: int | None) -> Answer:
    printer = _StreamPrinter()
    try:
        for event in rag.ask_stream(question, top_k):
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
    top_k: int = typer.Option(None, help="Number of chunks to retrieve."),
    stream: bool = typer.Option(
        True, "--stream/--no-stream", help="Print the answer as it is generated."
    ),
):
    """Ask a question about the indexed documents."""
    rag = RAG.from_settings()
    try:
        if stream:
            answer = _ask_streaming(rag, question, top_k)
        else:
            answer = rag.ask(question, top_k)
            typer.echo(_printable(answer.answer).strip())
    except GenerationError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    _print_sources(answer.citations)


@app.command("list")
def list_documents():
    """List documents indexed in the local store."""
    documents = RAG.from_settings().store.documents()
    for document in documents:
        typer.echo(f"{document.doc_id}  {document.source}  ({document.chunks} chunks)")


@app.command()
def delete(doc_id: str):
    """Delete an indexed document by its ID."""
    store = RAG.from_settings().store
    if not store.delete(doc_id):
        typer.secho(f"Document not found: {doc_id}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    typer.echo(f"Deleted document: {doc_id}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("citerag.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
