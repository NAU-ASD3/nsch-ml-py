# Contributing to nsch-ml

Thanks for your interest in contributing. This document covers the development setup, conventions, and PR workflow. It assumes familiarity with Python, Git, and GitHub PRs.

## Setup

Prerequisites:

- [`uv`](https://docs.astral.sh/uv/) (the project's package manager)
- Git

```bash
git clone https://github.com/NAU-ASD3/nsch-ml-py
cd nsch-ml-py
uv sync --group dev
uv run pre-commit install
```

`uv sync` installs all runtime and development dependencies into a project-local virtual environment at `.venv/`. You do not need to activate it; `uv run <command>` handles that.

## Running things

```bash
uv run pytest                              # full test suite
uv run pytest tests/test_soak.py           # a single file
uv run pytest -k "assign_folds"            # by test name pattern
uv run pytest --cov=src/nsch_ml            # with coverage
uv run pytest -m "not network"             # skip fixture-download tests
uv run ruff check .                        # lint
uv run ruff format --check .               # format check
uv run ruff format .                       # auto-format
uv run mypy src/                           # type check
uv run mkdocs serve                        # docs site on localhost:8000
```

The full local CI mirror (run before opening a PR):

```bash
uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy src/
```

A green local run is the gate for opening a PR.

## Code style

- **Type hints on every public signature.** `mypy --strict` must pass. Distinguish `pl.LazyFrame`, `pl.DataFrame`, and `pl.Series` in signatures; annotate the pandas boundary explicitly.
- **NumPy-style docstrings on every public function.** Include parameter descriptions, return type, and at least one `>>> ` doctest example for non-trivial functions.
- **`pathlib.Path` for paths**, never `str`. Functions that accept paths annotate them as `Path | str` and `Path(...)` them internally.
- **`snake_case`** for functions, variables, and modules.
- **`logging.getLogger(__name__)`** for diagnostics, never `print()`.
- **`Enum` / `IntEnum`** for finite sets of named values. No magic numbers in function bodies.
- **Imports are sorted by ruff.** Don't hand-curate import order.
- **Comment generously, especially the "why."** A comment that explains a non-obvious choice (a quirk of the original R analysis, a glmnet-versus-sklearn difference) is more valuable than one that restates the code.

## Architecture invariants

These are not preferences. Violations are bugs:

1. **Equivalence is judged on held-out predictions, not coefficients.** Any change that moves the port's predictions relative to the R reference must be justified against the tolerance in the planning docs. See `docs/design-decisions.md`.
2. **The one-hot encoder is pinned to the R model matrix.** The conditional reference-level rule must reproduce the committed golden header exactly, column set and per-variable reference level both. Never adjust the encoder without updating the golden test deliberately and saying so in the PR.
3. **Seeds are explicit parameters, never hidden global state.** The original analysis used different seeds per driver (core 1, fairness 42, clustering swept 1–10). Every function that randomizes takes a seed argument and documents it.
4. **Data prep stays in Polars; pandas exists only at the scikit-learn/XGBoost boundary.** No pandas upstream of the model matrix.
5. **No mutation.** Every transform returns a new frame or array.
6. **Fixtures follow the two-marker scheme.** Tests that download the Zenodo fixture carry `@pytest.mark.network` and verify the pinned checksum; tests that need the real harmonized NSCH dataset carry `@pytest.mark.integration` and are env-gated. Everything else runs on small synthetic data committed to the repo.
7. **Functions are small and single-purpose.** Aim for under 50 lines.

## Tests

- Plain `assert`, no assertion helpers.
- Compare full vectors and full frames, never spot-checked elements.
- Synthetic data constructed inside the test or its fixture, small enough to verify by eye.
- One behavior per test, named descriptively.

## PR workflow

Stacked PRs in dependency order. Each PR:

- Branches from `main` (or from the parent stacked PR's branch).
- Bumps the version in `pyproject.toml` (date format `2026.M.DD`, the date being the merge day).
- Adds an entry to `CHANGELOG.md` referencing the PR number.
- Includes complete tests, complete docstrings, and an mkdocs nav entry if it adds public API.
- Passes the full local CI mirror before review is requested.

For stacked PRs, the first line of the PR description must be: `⚠️ Stacked PR — branches from <parent_branch> (#<parent_PR>). Only files listed below are new; others belong to parent PR.`, followed by an explicit list of new files.

Squash-and-merge only. Reviewers are listed in `CODEOWNERS`.
