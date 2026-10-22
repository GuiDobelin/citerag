# Contributing

Thanks for helping out. Bug reports, ideas and pull requests are all welcome.

1. Open an issue first for anything bigger than a small fix, so we can agree on the approach.
2. Fork, create a branch, and set up the dev environment:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. Add tests for your change. Tests must run offline: use `HashEmbedder` and a fake LLM client
   (see `tests/test_pipeline.py`).
4. Make sure `pytest`, `ruff check .` and `ruff format --check .` pass.
5. Open a pull request describing what changed and why.

Good first contributions are listed under the
[`good first issue`](https://github.com/GuiDobelin/citerag/labels/good%20first%20issue) label.
