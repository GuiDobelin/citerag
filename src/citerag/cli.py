from pathlib import Path

import typer

from citerag.llm import GenerationError
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


@app.command()
def ask(question: str, top_k: int = typer.Option(None, help="Number of chunks to retrieve.")):
    """Ask a question about the indexed documents."""
    try:
        answer = RAG.from_settings().ask(question, top_k)
    except GenerationError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(answer.answer)
    if answer.citations:
        typer.echo("\nSources:")
        for i, c in enumerate(answer.citations, start=1):
            where = f"{c.source}, p. {c.page}" if c.page else c.source
            quote = " ".join(c.quote.split())[:160]
            typer.echo(f'  [{i}] {where}: "{quote}"')


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("citerag.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
