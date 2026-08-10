# Contributing to nsch-ml

This covers setup, conventions, and the PR workflow. It assumes you know
Python, Git, and GitHub pull requests.

## Setup

You need [`uv`](https://docs.astral.sh/uv/) and Git. Nothing else.

```bash
git clone https://github.com/NAU-ASD3/nsch-ml-py
cd nsch-ml-py
uv sync --group dev
uv run pre-commit install
```

`uv sync` builds a project-local environment at `.venv/`. Do not activate it;
`uv run <command>` handles that for you.

## The gate

One command. It has to pass before you open a PR, and it is what CI runs.

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q
```

Note that `mypy` takes no path argument. Its `files` setting in
`pyproject.toml` covers `src/nsch_ml`, `analyses`, and `tests`, and passing a
path on the command line silently overrides that, which means checking less
than CI does. The pre-commit hooks call the same argument-free commands
through `uv run`, so the hooks, the gate, and CI are the same programs
reading the same configuration.

If only `ruff format --check` complains, run `uv run ruff format .` and move
on. Do not hand-match its line breaking. A `ruff check` failure is worth
reading, since those catch real problems.

## Other commands

```bash
uv run pytest tests/test_soak.py           # one file
uv run pytest -k "assign_folds"            # by name pattern
uv run pytest --cov=src/nsch_ml            # with coverage
uv run pytest -m "not network"             # skip fixture downloads
uv run mkdocs serve                        # docs on localhost:8000
uv run pre-commit run --all-files          # hooks outside a commit
```

## Code style

- **Type hints on every public signature.** `mypy --strict` must pass.
  Distinguish `pl.LazyFrame`, `pl.DataFrame`, and `pl.Series`; annotate the
  pandas boundary explicitly.
- **NumPy-style docstrings on every public function**, with parameter
  descriptions, return type, and a `>>>` doctest for anything non-trivial.
- **`pathlib.Path` for paths**, never `str`. Accept `Path | str` and
  `Path(...)` internally.
- **`snake_case`** for functions, variables, and modules.
- **`logging.getLogger(__name__)`** in library code, never `print()`. Scripts
  in `analyses/` are the exception; see below.
- **`Enum` or `IntEnum`** for finite sets of named values. No magic numbers
  in function bodies.
- **Imports are sorted by ruff.** Do not hand-curate them.
- **Comment the why, not the what.** A note explaining a quirk of the
  original R analysis, or a glmnet-versus-scikit-learn difference, is worth
  more than a line restating the code beside it.

### Names

Names describe the value they hold. Single letters are reserved for loop
indices over a range and for the two established abbreviations `df` and `ax`.
Everything else gets a word: `distance` rather than `d`, `n_pairs` rather
than `n`, `train_source` rather than `src`.

A name used twice in one function must mean the same thing both times.
Reusing a short name for two purposes is how `ypos` ended up as both an array
and a list, and how `denom` ended up as both an integer and a string. Both
survived review and were caught by mypy rather than by anyone reading.

### Narrowing types from Polars

Polars reducers return a union spanning numeric, temporal, and string types
plus `None`, because what `.mean()` means depends on the dtype. `mypy
--strict` cannot narrow that to `float`, so `float(series.mean())` fails type
checking even when the column is plainly numeric.

Three ways out, in order of preference.

**`typing.cast`** when the surrounding code guarantees the dtype: the column
was just built with `.cast(pl.Int8)`, or it came out of arithmetic. Costs
nothing at runtime and reads as what it is, an assertion about a type you
already know. Quote the target, or ruff's `TC006` will object to evaluating
an annotation for no reason.

```python
prevalence = cast("float", frame["truth"].mean())
```

**An explicit `None` check** when the value comes from data you did not
construct, such as a user-supplied column or a frame that might be empty. It
is the only option that survives `python -O` and the only one that can carry
a useful message.

```python
value = frame[column].mean()
if value is None:
    raise ValueError(f"{column} is empty")
prevalence = float(value)
```

**`assert isinstance(...)`** only for genuine invariants, where a failure
means a bug in our code rather than bad input. Assertions are stripped under
`-O`, so never use one as a data check.

Do not reach for `.to_numpy()` to sidestep the typing. It materialises the
whole column to produce one number, which is invisible on 60 rows and
expensive on a million. Converting to numpy because you then want numpy
operations is a different thing and is fine.

## The `analyses/` directory

`analyses/` holds scripts, not library code. They are run by hand against
data that is not in the repository, they print their results rather than
logging them, and they are not imported by the package. Ruff allows `print`
there and nowhere else.

They are still type-checked and still formatted, and they still go through
review. A script that produces a number someone will quote is not a scratch
file.

Two habits that have earned their place:

**Verify before you write.** A script that transforms reference data should
check its output against something external before saving it. Recovering R's
coefficients was checked by reconstructing R's own AUC; repairing R's
predictions was checked against both the AUC and the outcome prevalence.
Without a check like that a subtly wrong transform looks exactly like a
correct one.

**Refuse to write partial results.** If a run is supposed to produce 60
splits and produces 20, stop rather than saving. A partial file compared
against a complete one looks like a real difference, and that has cost this
project time already.

Tests for pure helpers in `analyses/` live in `tests/test_analyses.py`, which
loads the scripts by path since `analyses/` is not a package.

## Architecture invariants

These are not preferences. Violations are bugs.

1. **Equivalence is judged on held-out predictions, not coefficients.** The
   margins are fixed in `docs/equivalence-margin.md` and were committed
   before the comparison was written. Any change that moves the port's
   predictions relative to the R reference has to be argued against them.
2. **The one-hot encoder is pinned to the R model matrix.** The conditional
   reference-level rule must reproduce the committed golden header exactly,
   both the column set and the per-variable reference level. Never adjust the
   encoder without updating the golden test deliberately and saying so in the
   PR.
3. **Seeds are explicit parameters, never hidden global state.** The original
   analysis used different seeds per driver: 1 for the core run, 42 for
   fairness, 1 through 10 swept for clustering. Every function that
   randomises takes a seed argument and documents it.
4. **Data prep stays in Polars.** pandas exists only at the scikit-learn and
   XGBoost boundary, never upstream of the model matrix.
5. **No mutation.** Every transform returns a new frame or array.
6. **Fixtures follow the two-marker scheme.** Tests that download the Zenodo
   fixture carry `@pytest.mark.network` and verify the pinned checksum; tests
   needing the real harmonised NSCH dataset carry `@pytest.mark.integration`
   and are environment-gated. Everything else runs on small synthetic data
   committed to the repository.
7. **Functions are small and single-purpose.** Under 50 lines is the target.

## Tests

- Plain `assert`. No assertion helpers.
- Compare full vectors and full frames, never spot-checked elements.
- Synthetic data built inside the test or its fixture, small enough to check
  by eye.
- One behaviour per test, named for the behaviour.

Coverage is measured on `src/nsch_ml` only, with a floor of 85%. The
`analyses/` scripts are excluded because they run against data that is not in
the repository, so a coverage figure for them would mean nothing.

A test that encodes a convention discovered the hard way is worth more than
one that restates an obvious property. `test_find_clusters_groups_identical_runs_without_shattering`
exists because a zero distance once split every run into its own cluster.

## PR workflow

Stacked PRs in dependency order. Every PR:

- Branches from `main`, or from the parent stacked PR's branch.
- Bumps `version` in `pyproject.toml` to the merge date, format `2026.M.DD`.
  Same-day collisions get a micro suffix in merge order. Run `uv lock` and
  stage `uv.lock` in the same `git add` as `pyproject.toml`; staging it
  afterwards makes pre-commit rewrite it and fail the commit.
- Adds a `CHANGELOG.md` entry under a `## YYYY.M.DD (PR#NN)` heading.
  One change per bullet. Say what changed and whether it affects anyone;
  leave the numbers to the documents that own them.
- Includes tests, docstrings, and an mkdocs nav entry if it adds public API.
- Passes the gate before review is requested.

For a stacked PR, the first line of the description must be:

> ⚠️ Stacked PR — branches from `<parent_branch>` (#`<parent_PR>`). Only files
> listed below are new; others belong to parent PR.

followed by an explicit list of the new files. Reviewers work through several
of these at a time and a duplicated diff wastes their attention.

Branch protection requires a pull request and one approving review, and
forbids force-pushing to `main`. Squash-and-merge only. Reviewers are in
`CODEOWNERS`.

Because merges are squashed, a branch cut from an already-merged branch will
not share history with `main` and its PR will re-show the whole parent diff.
Always branch from an up-to-date `main`.

## Where things live

|                                   |                                                |
| --------------------------------- | ---------------------------------------------- |
| `src/nsch_ml/`                    | the library                                    |
| `analyses/`                       | run-by-hand scripts                            |
| `tests/`                          | tests, including for `analyses/` helpers       |
| `docs/replication-equivalence.md` | what matches the R analysis, what differs, why |
| `docs/equivalence-margin.md`      | the pass/fail standard, fixed in advance       |
| `CHANGELOG.md`                    | what changed and when                          |

Numbers live in the documents, not in docstrings. A figure embedded in a
module docstring goes stale the first time anything is rerun, and nobody
notices. Point at the document instead.
