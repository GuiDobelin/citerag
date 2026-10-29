"""Render the README terminal demo (docs/demo.svg) from a real `citerag` session.

The session runs offline and needs no API key: the extractive generator and the hashing
embedder are deterministic, so the SVG only changes when the CLI output does.

    python scripts/make_demo.py            # writes docs/demo.svg
    python scripts/make_demo.py out.svg
"""

import os
import sys
import tempfile
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

from typer.testing import CliRunner

from citerag.cli import app

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "docs" / "demo"
OUTPUT = ROOT / "docs" / "demo.svg"
QUESTION = "How long do customers have to ask for a refund?"

COLUMNS = 92
LINE_HEIGHT = 20
CHAR_WIDTH = 8.4
PADDING = 18
TITLE_BAR = 30
# Seconds before each kind of line appears, to mimic typing a command and a streamed answer.
COMMAND_DELAY = 0.9
OUTPUT_DELAY = 0.25


def run_session(data_dir: Path) -> list[tuple[str, str]]:
    """Run the demo commands and return (command, output) pairs."""
    env = {
        "CITERAG_GENERATOR": "extractive",
        "CITERAG_EMBEDDER": "hash",
        "CITERAG_STORE": "memory",
        "CITERAG_DATA_DIR": str(data_dir),
        "CITERAG_CHUNK_SIZE": "160",
        "CITERAG_CHUNK_OVERLAP": "0",
        "CITERAG_TOP_K": "2",
    }
    runner = CliRunner()
    samples = sorted(p.name for p in SAMPLES.glob("*.md"))
    steps = [
        ("citerag ingest " + " ".join(samples), ["ingest", *samples]),
        (f'citerag ask "{QUESTION}"', ["ask", QUESTION]),
    ]
    session = []
    cwd = Path.cwd()
    os.chdir(SAMPLES)  # relative paths in the output, and no stray .env from the caller
    try:
        for shown, args in steps:
            result = runner.invoke(app, args, env=env, color=False)
            if result.exit_code != 0:
                raise RuntimeError(f"`{shown}` failed:\n{result.output}") from result.exception
            session.append((shown, result.stdout))
    finally:
        os.chdir(cwd)
    return session


def _wrap(line: str) -> list[str]:
    if len(line) <= COLUMNS:
        return [line]
    indent = " " * (len(line) - len(line.lstrip()) + 4)
    return textwrap.wrap(line, COLUMNS, subsequent_indent=indent, break_on_hyphens=False)


def render_svg(session: list[tuple[str, str]]) -> str:
    """Draw the session as a terminal window whose lines appear one after another."""
    rows: list[tuple[str, str, float]] = []  # (css class, text, delay in seconds)
    delay = 0.3
    for command, output in session:
        rows.append(("cmd", f"$ {command}", delay))
        delay += COMMAND_DELAY
        for raw in output.rstrip("\n").split("\n"):
            for line in _wrap(raw):
                cls = "head" if line == "Sources:" else "out"
                rows.append((cls, line, delay))
                delay += OUTPUT_DELAY
        rows.append(("out", "", delay))
        delay += OUTPUT_DELAY
    rows.pop()  # no blank line after the last command

    width = round(PADDING * 2 + COLUMNS * CHAR_WIDTH)
    height = TITLE_BAR + PADDING * 2 + LINE_HEIGHT * len(rows)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-label="citerag ingesting two files and streaming a cited answer">',
        "<style>",
        "text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
        "font-size:14px;white-space:pre;fill:#d4d4d4}",
        ".cmd{fill:#7ee787}.head{fill:#79c0ff;font-weight:bold}",
        ".l{opacity:0;animation:show .01s linear forwards}",
        "@keyframes show{to{opacity:1}}",
        "@media (prefers-reduced-motion:reduce){.l{animation:none;opacity:1}}",
        "</style>",
        f'<rect width="{width}" height="{height}" rx="8" fill="#0d1117"/>',
        f'<rect width="{width}" height="{TITLE_BAR}" rx="8" fill="#161b22"/>',
        '<circle cx="18" cy="15" r="6" fill="#ff5f57"/>',
        '<circle cx="38" cy="15" r="6" fill="#febc2e"/>',
        '<circle cx="58" cy="15" r="6" fill="#28c840"/>',
    ]
    for n, (cls, text, at) in enumerate(rows, start=1):
        y = TITLE_BAR + PADDING + LINE_HEIGHT * n - 5
        lines.append(
            f'<text class="l {cls}" x="{PADDING}" y="{y}" '
            f'style="animation-delay:{at:.2f}s">{escape(text)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> Path:
    args = sys.argv[1:] if argv is None else argv
    out = Path(args[0]) if args else OUTPUT
    with tempfile.TemporaryDirectory() as tmp:
        svg = render_svg(run_session(Path(tmp)))
    out.write_text(svg, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(f"wrote {main()}")
