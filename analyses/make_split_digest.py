"""Verify the Python SOAK splitter against the R mlr3resampling fixture.

Reads the archived ``ResamplingSameOtherSizesCV`` instance from
``soak_fixture/`` (folds=10, sizes=0, seed=1, 46,010 rows over survey
years 2019 and 2020), replays it through :mod:`nsch_ml.soak`, and
reports agreement in three tiers:

1. STRUCTURAL  -- the set of 100 (test subset, train source, fold,
   downsampled) iteration keys matches R exactly.
2. FULL SPLITS -- for the 60 non-downsampled iterations, both the test
   and train row sets match R index-for-index.
3. DOWNSAMPLED -- for the 40 downsampled iterations, the test rows
   match exactly and the train set's size and per-outcome-stratum
   counts match. Row *membership* is not expected to match: R's
   ``sample()`` and NumPy's generator are different RNGs, so the rule
   is replicated but the draw is not.

Run from the repository root::

    uv run python analyses/make_split_digest.py

Exits non-zero if any tier fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from nsch_ml.soak import assign_folds, iter_soak_splits

# --- Fixture location -----------------------------------------------------
# The fixture lives outside the repository (it is ~1.3 GB extracted and is
# never committed). Override with NSCH_SOAK_FIXTURE if yours sits elsewhere.
DEFAULT_FIXTURE = Path.home() / "Documents/NAU/Grad/Research/ADSI/soak_fixture"

N_FOLDS = 10
SEED = 1
EXPECTED_ROWS = 46010
EXPECTED_ITERATIONS = 100


def fixture_dir() -> Path:
    import os

    return Path(os.environ.get("NSCH_SOAK_FIXTURE", DEFAULT_FIXTURE))


def load_fixture(root: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (folds, iterations_long), validated for shape and alignment."""
    folds = pl.read_csv(root / "nsch_autism_folds.csv")
    # Only two columns are needed from the 300-column design matrix.
    design = pl.read_csv(
        root / "data_Classif" / "NSCH_autism.csv",
        columns=["survey_year", "y"],
    )
    iterations = pl.read_csv(root / "nsch_autism_iterations_long.csv")

    if folds.height != EXPECTED_ROWS:
        msg = f"folds has {folds.height} rows, expected {EXPECTED_ROWS}"
        raise ValueError(msg)
    if design.height != EXPECTED_ROWS:
        msg = f"design matrix has {design.height} rows, expected {EXPECTED_ROWS}"
        raise ValueError(msg)

    # row_id must be exactly 1..N in order, so positional index i in the
    # design matrix corresponds to row_id i+1 in the fold/iteration files.
    row_id = folds["row_id"].to_numpy()
    if not np.array_equal(row_id, np.arange(1, EXPECTED_ROWS + 1)):
        msg = "folds row_id is not exactly 1..N in order; alignment unsafe"
        raise ValueError(msg)

    # Independent cross-check that the two files describe the same rows.
    mismatch = int(
        (folds["test.subset"].cast(pl.Utf8) != design["survey_year"].cast(pl.Utf8)).sum()
    )
    if mismatch:
        msg = f"{mismatch} rows disagree between folds test.subset and design survey_year"
        raise ValueError(msg)

    folds = folds.with_columns(design["y"].alias("y"))
    return folds, iterations


def r_iterations(iterations: pl.DataFrame) -> dict[tuple[str, str, int, bool], dict[str, set[int]]]:
    """Collapse the long iteration table into {key: {role: row_id set}}."""
    grouped = iterations.group_by(
        ["test.subset", "train.subsets", "test.fold", "downsampled", "role"]
    ).agg(pl.col("row_id"))

    out: dict[tuple[str, str, int, bool], dict[str, set[int]]] = {}
    for subset, source, fold, down, role, ids in grouped.iter_rows():
        key = (str(subset), str(source), int(fold), bool(down))
        out.setdefault(key, {})[str(role)] = set(ids)
    return out


def py_iterations(folds: pl.DataFrame) -> dict[tuple[str, str, int, bool], dict[str, set[int]]]:
    """Run the Python splitter and shape its output the same way.

    Indices are converted from 0-based positions to 1-based row_ids so
    the two sides are directly comparable.
    """
    fold_ids = assign_folds(
        subset=folds["test.subset"],
        outcome=folds["y"],
        n_folds=N_FOLDS,
        precomputed=folds["fold"],
    )
    out: dict[tuple[str, str, int, bool], dict[str, set[int]]] = {}
    for split in iter_soak_splits(
        fold_ids=fold_ids,
        subset=folds["test.subset"],
        outcome=folds["y"],
        sizes=0,
        seed=SEED,
    ):
        key = (
            split.test_subset,
            split.train_source.value,
            split.fold,
            split.downsampled,
        )
        out[key] = {
            "test": set((split.test_idx + 1).tolist()),
            "train": set((split.train_idx + 1).tolist()),
        }
    return out


def stratum_counts(row_ids: set[int], outcome: np.ndarray) -> dict[str, int]:
    """Per-outcome-level counts for a set of 1-based row_ids."""
    idx = np.fromiter(row_ids, dtype=np.int64) - 1
    levels, counts = np.unique(outcome[idx], return_counts=True)
    return {str(k): int(v) for k, v in zip(levels, counts, strict=True)}


def main() -> int:
    root = fixture_dir()
    print(f"fixture: {root}")
    if not root.is_dir():
        print("FAIL: fixture directory not found", file=sys.stderr)
        return 2

    folds, iterations = load_fixture(root)
    outcome = folds["y"].to_numpy()
    print(f"rows: {folds.height}  subsets: {sorted(folds['test.subset'].unique().to_list())}")

    r_it = r_iterations(iterations)
    py_it = py_iterations(folds)

    failures: list[str] = []

    # --- Tier 1: structural -------------------------------------------
    print("\n== Tier 1: structural ==")
    r_keys, py_keys = set(r_it), set(py_it)
    print(f"R iterations:      {len(r_keys)}")
    print(f"Python iterations: {len(py_keys)}")
    only_r, only_py = sorted(r_keys - py_keys), sorted(py_keys - r_keys)
    if only_r or only_py:
        failures.append("structural key mismatch")
        for k in only_r[:10]:
            print(f"  only in R:      {k}")
        for k in only_py[:10]:
            print(f"  only in Python: {k}")
        print("STRUCTURAL: FAIL")
    else:
        expected = len(r_keys) == EXPECTED_ITERATIONS
        print(f"keys identical, count == {EXPECTED_ITERATIONS}: {expected}")
        print("STRUCTURAL: PASS" if expected else "STRUCTURAL: FAIL (unexpected count)")
        if not expected:
            failures.append("iteration count")

    shared = sorted(r_keys & py_keys)

    # --- Tier 2: full splits ------------------------------------------
    print("\n== Tier 2: full splits (index-for-index) ==")
    full = [k for k in shared if not k[3]]
    bad_full = []
    for key in full:
        r_side, py_side = r_it[key], py_it[key]
        if r_side.get("test", set()) != py_side["test"]:
            bad_full.append((key, "test"))
        elif r_side.get("train", set()) != py_side["train"]:
            bad_full.append((key, "train"))
    print(
        f"compared: {len(full)}   exact: {len(full) - len(bad_full)}   mismatched: {len(bad_full)}"
    )
    for key, which in bad_full[:10]:
        r_n = len(r_it[key].get(which, set()))
        py_n = len(py_it[key][which])
        print(f"  {key} {which}: R n={r_n}, Python n={py_n}")
    if bad_full:
        failures.append("full splits")
    print("FULL SPLITS: PASS" if not bad_full else "FULL SPLITS: FAIL")

    # --- Tier 3: downsampled ------------------------------------------
    print("\n== Tier 3: downsampled (test exact; train counts only) ==")
    down = [k for k in shared if k[3]]
    bad_test, bad_size, bad_strata = [], [], []
    for key in down:
        r_side, py_side = r_it[key], py_it[key]
        if r_side.get("test", set()) != py_side["test"]:
            bad_test.append(key)
        r_train, py_train = r_side.get("train", set()), py_side["train"]
        if len(r_train) != len(py_train):
            bad_size.append((key, len(r_train), len(py_train)))
        elif stratum_counts(r_train, outcome) != stratum_counts(py_train, outcome):
            bad_strata.append(key)
    print(f"compared: {len(down)}")
    print(f"  test rows exact:      {len(down) - len(bad_test)}/{len(down)}")
    print(f"  train size match:     {len(down) - len(bad_size)}/{len(down)}")
    print(f"  stratum count match:  {len(down) - len(bad_size) - len(bad_strata)}/{len(down)}")
    for key, r_n, py_n in bad_size[:10]:
        print(f"  {key}: R n={r_n}, Python n={py_n}, delta={py_n - r_n}")
    if bad_test or bad_size or bad_strata:
        failures.append("downsampled")
    print("DOWNSAMPLED: PASS" if not (bad_test or bad_size or bad_strata) else "DOWNSAMPLED: FAIL")

    # --- Verdict ------------------------------------------------------
    print("\n== Verdict ==")
    if failures:
        print("FAIL: " + ", ".join(failures))
        return 1
    print("PASS: all three tiers agree with the R fixture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
