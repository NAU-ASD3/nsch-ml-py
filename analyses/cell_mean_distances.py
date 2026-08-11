"""How far apart are these runs when fold numbering is set aside?

The existing clustering in `analyses/r_vs_r.py` measures the mean absolute
difference in AUC between two runs, fold by fold. That is the sharper
comparison and it is only available when both runs assign the same children
to the same fold number.

The published results in `data_Classif_batchmark_registry.csv` do not meet
that condition. They come from `same_other_cv`, the resampling class that has
since been removed, while every run we have used `same_other_sizes_cv`. Their
per-fold AUCs correlate at about -0.10 as numbered and 0.81 to 0.98 when both
sides are sorted, so the two analyses saw a similar spread of folds under
unrelated numbering.

Pairing folds across that boundary would produce a number in the same units
as the per-fold distances and mean something entirely different, which is the
kind of error that survives review because the output looks the same. So this
script uses a quantity that does not depend on fold numbering at all: the
mean AUC within each of the six (test subset, train source) cells.

Cell means are what the paper reports and what the SOAK conclusions rest on.
Two runs that agree on them agree about the analysis, whatever their folds
did individually.

Reading the two together is the point. If runs cluster under the per-fold
metric while agreeing on cell means, then whatever separates them affects
individual fits without moving aggregate performance, and the published
conclusions are robust to it.

Run from the repository root::

    uv run python analyses/cell_mean_distances.py \\
        --reproduce-dir $REPRO --registry $PAPER/data_Classif_batchmark_registry.csv
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

DEFAULT_REPRODUCE = Path.home() / "Documents/NAU/Grad/Research/ADSI/reproduce-soak-nsch"
TRAIN_SOURCES = ("same", "other", "all")
CELL_KEY = ["test_subset", "train_source"]
EXPECTED_CELLS = 6


def cell_means(per_fold: pl.DataFrame) -> pl.DataFrame:
    """Mean AUC in each (test subset, train source) cell, sorted for alignment."""
    return (
        per_fold.group_by(CELL_KEY)
        .agg(pl.col("auc").mean().alias("mean_auc"), pl.len().alias("n_folds"))
        .sort(CELL_KEY)
    )


def load_reference(path: Path) -> pl.DataFrame | None:
    """One of our R runs, keeping the full split in each cell."""
    try:
        frame = pl.read_csv(path, infer_schema_length=20000)
    except Exception as error:
        print(f"  skip {path.name}: unreadable ({error})")
        return None
    required = ["test.subset", "train.subsets", "test.fold", "n.train.groups", "classif.auc"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        print(f"  skip {path.name}: missing {missing}")
        return None
    if "learner_id" in frame.columns:
        frame = frame.filter(pl.col("learner_id") == "classif.cv_glmnet")
    if frame.height == 0:
        print(f"  skip {path.name}: no classif.cv_glmnet rows")
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
        .unique(subset=["test_subset", "train_source", "fold"], keep="first")
    )


def load_registry(path: Path) -> pl.DataFrame:
    """The published rows, whose columns are named differently."""
    frame = pl.read_csv(path, infer_schema_length=20000).filter(
        (pl.col("task_id") == "NSCH_autism") & (pl.col("learner_id") == "classif.cv_glmnet")
    )
    return frame.select(
        pl.col("test.group").cast(pl.Utf8).alias("test_subset"),
        pl.col("train.groups").alias("train_source"),
        pl.col("test.fold").alias("fold"),
        pl.col("classif.auc").alias("auc"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reproduce-dir", default=str(DEFAULT_REPRODUCE))
    parser.add_argument("--registry", default=None)
    args = parser.parse_args()

    results_dir = Path(os.environ.get("NSCH_REPRODUCE_DIR", args.reproduce_dir)) / "results"
    if not results_dir.is_dir():
        print(f"FAIL: {results_dir} not found", file=sys.stderr)
        return 2

    print("runs")
    runs: dict[str, pl.DataFrame] = {}
    for path in sorted(results_dir.rglob("*.csv")):
        if path.stem.startswith("meta") or "jobs" in path.stem or "coefficients" in path.stem:
            continue
        if "predictions" in path.stem:
            continue
        frame = load_reference(path)
        if frame is None:
            continue
        name = f"{path.parent.name}/{path.stem}"
        runs[name] = cell_means(frame)
        print(f"  {name:<38} {frame.height:>4} splits")

    if args.registry:
        registry_path = Path(args.registry)
        if not registry_path.is_file():
            print(f"FAIL: registry not found at {registry_path}", file=sys.stderr)
            return 2
        registry = load_registry(registry_path)
        runs["PUBLISHED/registry"] = cell_means(registry)
        print(f"  {'PUBLISHED/registry':<38} {registry.height:>4} splits  (same_other_cv)")

    if len(runs) < 2:
        print("need at least two runs", file=sys.stderr)
        return 1
    for name, means in runs.items():
        if means.height != EXPECTED_CELLS:
            print(
                f"FAIL: {name} has {means.height} cells, expected {EXPECTED_CELLS}", file=sys.stderr
            )
            return 1

    # ------------------------------------------------------- the cells
    print("\n" + "=" * 96)
    print("Mean AUC per cell")
    print("=" * 96)
    reference_cells = runs[next(iter(runs))].select(CELL_KEY)
    header = "".join(f"{name.split('/')[-1][:13]:>15}" for name in runs)
    print(f"{'subset':>8}{'source':>8}{header}")
    for cell in reference_cells.iter_rows(named=True):
        values = []
        for means in runs.values():
            row = means.filter(
                (pl.col("test_subset") == cell["test_subset"])
                & (pl.col("train_source") == cell["train_source"])
            )
            values.append(float(row["mean_auc"][0]))
        formatted = "".join(f"{value:>15.6f}" for value in values)
        print(f"{cell['test_subset']:>8}{cell['train_source']:>8}{formatted}")

    # -------------------------------------------------- pairwise table
    print("\n" + "=" * 96)
    print("Pairwise distance, mean absolute difference across the six cell means")
    print("=" * 96)
    print(f"{'run A':<30}{'run B':<30}{'distance':>12}")
    names = list(runs)
    distances: dict[tuple[str, str], float] = {}
    for name_a, name_b in itertools.combinations(names, 2):
        merged = runs[name_a].join(runs[name_b], on=CELL_KEY, how="inner", suffix="_b")
        if merged.height != EXPECTED_CELLS:
            print(f"{name_a:<30}{name_b:<30}{'cells do not align':>12}")
            continue
        distance = float(
            np.abs(merged["mean_auc"].to_numpy() - merged["mean_auc_b"].to_numpy()).mean()
        )
        distances[(name_a, name_b)] = distance
        print(f"{name_a:<30}{name_b:<30}{distance:>12.6f}")

    # -------------------------------------------------------- summary
    print("\n" + "=" * 96)
    print("Reading this against the per-fold distances")
    print("=" * 96)
    gaps = np.array(list(distances.values()))
    print(f"  pairs compared            {len(gaps)}")
    print(f"  smallest cell-mean gap    {gaps.min():.6f}")
    print(f"  largest cell-mean gap     {gaps.max():.6f}")
    print(f"  median                    {np.median(gaps):.6f}")
    if args.registry:
        registry_gaps = {
            (a if b == "PUBLISHED/registry" else b): gap
            for (a, b), gap in distances.items()
            if "PUBLISHED/registry" in (a, b)
        }
        nearest = min(registry_gaps, key=lambda key: registry_gaps[key])
        furthest = max(registry_gaps, key=lambda key: registry_gaps[key])
        print()
        print(f"  published run, nearest    {nearest} at {registry_gaps[nearest]:.6f}")
        print(f"  published run, furthest   {furthest} at {registry_gaps[furthest]:.6f}")
        spread = max(registry_gaps.values()) - min(registry_gaps.values())
        print(f"  published run, spread     {spread:.6f}")
    print()
    print("  These distances are not comparable to the per-fold distances in")
    print("  r_vs_r.py. Averaging within a cell before differencing removes the")
    print("  fold-to-fold variation that the per-fold metric measures, so these")
    print("  numbers will be smaller for the same pair of runs. Compare them to")
    print("  each other, never across the two scripts.")
    print()
    print("  If the runs separate under the per-fold metric while agreeing here,")
    print("  then whatever divides them moves individual fits without moving")
    print("  aggregate performance, and the published conclusions are robust to it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
