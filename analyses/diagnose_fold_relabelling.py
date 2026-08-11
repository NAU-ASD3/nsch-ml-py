"""Is the February-to-August per-fold gap a relabelling artifact?

The per-fold metric in `analyses/r_vs_r.py` separates the R runs into groups
about 0.0073 apart. The cell means in `analyses/cell_mean_distances.py` show
no such separation: every pair sits between 0 and 0.00051, and pairs spanning
the supposed boundary are indistinguishable from pairs within it.

Both cannot be describing the same thing. Averaging within a cell removes
fold-to-fold variation, so a real difference in aggregate performance would
survive it. A difference in *which children land in which fold* would not,
because the cell mean covers all of them either way.

That suggests the per-fold gap may be an artifact of pairing fold 7 with fold
7 when the two runs put different children in fold 7. The fold assignments
themselves are not recoverable: the scores files hold per-fold AUC and nothing
about row membership, and only one run ever exported predictions. So the
question has to be answered from the AUC values alone.

The test. Within a cell, each run gives ten numbers. If the two runs share a
fold assignment, pairing by fold number is the right pairing and the mean
absolute difference is small. If one run's folds are a permutation of the
other's, pairing by number compares unrelated subsets and looks large, but
some permutation would make it small again. So: compute the distance under
the labelled pairing, then under the assignment that minimises it, and see
how far it falls.

The control. Optimal matching over ten values always reduces distance
somewhat, purely by chance, so the reduction means nothing on its own. Pairs
we are confident share a fold assignment give the chance floor: runs within
the February group, and runs within the August group, which differ only in
seed or machine. If the cross-boundary reduction is no larger than that
floor, the relabelling explanation fails and the gap is real. If it is much
larger, the folds differ and the per-fold comparison across that boundary was
never meaningful.

Run from the repository root::

    uv run python analyses/test_fold_relabelling.py --reproduce-dir $REPRO
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

DEFAULT_REPRODUCE = Path.home() / "Documents/NAU/Grad/Research/ADSI/reproduce-soak-nsch"
CELL_KEY = ["test_subset", "train_source"]
EXPECTED_FOLDS = 10

# Runs believed to share a fold assignment, used as the chance floor. Within
# each group the runs differ only by machine, backend, or inner seed, none of
# which touches the outer fold assignment fixed by set.seed(1) in NSCH.R.
FEBRUARY_GROUP = {
    "NSCH_batchtools",
    "NSCH_local",
    "NSCH_local_desktop",
    "NSCH_local_laptop",
    "NSCH_mpi",
    "NSCH_proj",
}
AUGUST_GROUP = {"NSCH_seed1", "NSCH_seed2", "NSCH_seed3", "NSCH_unseeded"}


def load_run(path: Path) -> pl.DataFrame | None:
    try:
        frame = pl.read_csv(path, infer_schema_length=20000)
    except Exception:
        return None
    required = ["test.subset", "train.subsets", "test.fold", "n.train.groups", "classif.auc"]
    if [name for name in required if name not in frame.columns]:
        return None
    if "learner_id" in frame.columns:
        frame = frame.filter(pl.col("learner_id") == "classif.cv_glmnet")
    if frame.height == 0:
        return None
    return (
        frame.select(
            pl.col("test.subset").cast(pl.Utf8).alias("test_subset"),
            pl.col("train.subsets").alias("train_source"),
            pl.col("test.fold").alias("fold"),
            pl.col("n.train.groups"),
            pl.col("classif.auc").alias("auc"),
        )
        .sort("n.train.groups", descending=True)
        .unique(subset=[*CELL_KEY, "fold"], keep="first")
        .sort([*CELL_KEY, "fold"])
    )


def labelled_and_matched(left: pl.DataFrame, right: pl.DataFrame) -> tuple[float, float]:
    """Mean absolute difference under fold numbering, and under the best pairing."""
    labelled_gaps, matched_gaps = [], []
    cells = left.select(CELL_KEY).unique().sort(CELL_KEY)
    for cell in cells.iter_rows(named=True):
        selector = (pl.col("test_subset") == cell["test_subset"]) & (
            pl.col("train_source") == cell["train_source"]
        )
        left_values = left.filter(selector).sort("fold")["auc"].to_numpy()
        right_values = right.filter(selector).sort("fold")["auc"].to_numpy()
        if len(left_values) != EXPECTED_FOLDS or len(right_values) != EXPECTED_FOLDS:
            continue
        labelled_gaps.extend(np.abs(left_values - right_values).tolist())
        cost = np.abs(left_values[:, None] - right_values[None, :])
        rows, columns = linear_sum_assignment(cost)
        matched_gaps.extend(cost[rows, columns].tolist())
    return float(np.mean(labelled_gaps)), float(np.mean(matched_gaps))


def group_of(name: str) -> str:
    stem = name.split("/")[-1]
    if stem in FEBRUARY_GROUP:
        return "Feb"
    if stem in AUGUST_GROUP:
        return "Aug"
    return "?"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reproduce-dir", default=str(DEFAULT_REPRODUCE))
    args = parser.parse_args()

    results_dir = Path(os.environ.get("NSCH_REPRODUCE_DIR", args.reproduce_dir)) / "results"
    if not results_dir.is_dir():
        print(f"FAIL: {results_dir} not found", file=sys.stderr)
        return 2

    runs: dict[str, pl.DataFrame] = {}
    for path in sorted(results_dir.rglob("*.csv")):
        if path.stem.startswith("meta") or "jobs" in path.stem:
            continue
        if "coefficients" in path.stem or "predictions" in path.stem:
            continue
        frame = load_run(path)
        if frame is not None:
            runs[path.stem] = frame
    print(f"{len(runs)} runs: {', '.join(sorted(runs))}\n")
    if len(runs) < 2:
        print("need at least two runs", file=sys.stderr)
        return 1

    print("=" * 92)
    print("Distance under fold numbering, and under the pairing that minimises it")
    print("=" * 92)
    print(f"{'run A':<24}{'run B':<24}{'pair':>7}{'labelled':>11}{'matched':>11}{'drop':>9}")

    within: list[float] = []
    across: list[float] = []
    for name_a, name_b in itertools.combinations(sorted(runs), 2):
        labelled, matched = labelled_and_matched(runs[name_a], runs[name_b])
        if labelled == 0:
            continue
        drop = 1 - matched / labelled
        group_a, group_b = group_of(name_a), group_of(name_b)
        pair = f"{group_a}-{group_b}"
        if group_a == group_b and group_a != "?":
            within.append(drop)
        elif "?" not in (group_a, group_b):
            across.append(drop)
        print(f"{name_a:<24}{name_b:<24}{pair:>7}{labelled:>11.6f}{matched:>11.6f}{drop:>8.1%}")

    print("\n" + "=" * 92)
    print("Reading this")
    print("=" * 92)
    if within:
        print(f"  within a group, mean drop from optimal matching : {np.mean(within):>7.1%}")
        print(
            f"    range                                        : "
            f"{min(within):.1%} to {max(within):.1%}"
        )
    if across:
        print(f"  across the groups, mean drop                   : {np.mean(across):>7.1%}")
        print(
            f"    range                                        : "
            f"{min(across):.1%} to {max(across):.1%}"
        )
    print()
    if within and across:
        if np.mean(across) > np.mean(within) + 0.15:
            print("  Matching helps far more across the boundary than within it, which is")
            print("  what different fold assignments would produce. The per-fold distance")
            print("  across that boundary compares different children and should not be")
            print("  read as a difference in the analysis.")
        else:
            print("  Matching helps about as much across the boundary as within it, so the")
            print("  reduction is what chance alone gives. The fold assignments are not")
            print("  the explanation, and the per-fold gap reflects something real.")
    print()
    print("  Either way this is indirect. The fold assignments are not recorded in")
    print("  any February artifact, so the argument runs through the AUC values")
    print("  rather than through the assignments themselves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
