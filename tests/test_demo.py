import importlib.util
import re
from pathlib import Path
from xml.etree import ElementTree

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("make_demo", ROOT / "scripts" / "make_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lines(svg: str) -> list[str]:
    tree = ElementTree.fromstring(svg)
    return [el.text or "" for el in tree.iter("{http://www.w3.org/2000/svg}text")]


def test_session_runs_offline_and_cites_the_refund_policy(demo, tmp_path):
    (ingest, ingest_out), (ask, ask_out) = demo.run_session(tmp_path)
    assert ingest.startswith("citerag ingest ")
    assert ingest_out.count("indexed ") == 2
    assert "30 days" in ask_out
    assert "Sources:\n  [1] refund-policy.md" in ask_out


def test_render_escapes_text_and_reveals_lines_in_order(demo):
    svg = demo.render_svg([("echo '<a & b>'", "<a & b>\nSources:\n")])
    assert _lines(svg) == ["$ echo '<a & b>'", "<a & b>", "Sources:"]
    delays = [float(d) for d in re.findall(r"animation-delay:([\d.]+)s", svg)]
    assert delays == sorted(delays) and len(set(delays)) == len(delays)


def test_long_lines_are_wrapped_to_the_terminal_width(demo):
    long = "  [1] doc.md: " + "word " * 40
    lines = _lines(demo.render_svg([("x", long)]))[1:]
    assert len(lines) > 1
    assert all(len(line) <= demo.COLUMNS for line in lines)
    assert all(line.startswith("      ") for line in lines[1:])


def test_committed_demo_matches_the_current_cli_output(demo, tmp_path):
    out = demo.main([str(tmp_path / "demo.svg")])
    committed = (ROOT / "docs" / "demo.svg").read_text(encoding="utf-8")
    assert out.read_text(encoding="utf-8") == committed, "run: python scripts/make_demo.py"
