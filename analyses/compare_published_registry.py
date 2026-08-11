"""Can the published results be compared fold by fold, or only in aggregate?

`data_Classif_batchmark_registry.csv` in the cv-same-other-paper repository
holds the results the SOAK paper was written from. Its NSCH_autism rows
reproduce the two figures quoted in Section 4.3 exactly, so this is the
publication data rather than a rerun.

Comparing it to our reference runs per fold assumes both assigned the same
children to the same fold number. That is not obviously true. The registry
used `same_other_cv`, the class that has since been removed from
mlr3resampling; every run we have used `same_other_sizes_cv`. Our splitter
was verified against a fold column exported from a `sizes_cv` instance, which
says nothing about what the earlier class did.

Two ways to tell without needing the original fold assignments.

  Aggregate. Mean AUC per (test subset, train source) is invariant to how
  folds are numbered, so it is comparable either way. If the registry and a
  reference run disagree here, they are different analyses and nothing else
  matters.

  Per fold. If fold numbering matches, per-fold AUCs should track closely,
  and the correlation across the 10 folds within a cell should be high. If
  numbering is arbitrary between the two, the correlation will sit near zero
  while the sorted values still agree. Comparing sorted against unsorted
  separates those cases.

Run from the repository root::

    uv run python analyses/compare_published_registry.py \\
        --registry  $PAPER/data_Classif_batchmark_registry.csv \\
        --reference $REPRO/results/seed-variation/NSCH_seed1.csv

where $PAPER is the cv-same-other-paper checkout and $REPRO the
reproduce-soak-nsch one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

TRAIN_SOURCES = ("same", "other", "all")
PAPER_2020 = {"all": 0.9670, "same": 0.9658}


def load_registry(path: Path) -> pl.DataFrame:
    """The published rows for this task and learner, renamed to our columns."""
    frame = pl.read_csv(path, infer_schema_length=20000).filter(
        (pl.col("task_id") == "NSCH_autism") & (pl.col("learner_id") == "classif.cv_glmnet")
    )
    return frame.select(
        pl.col("test.group").cast(pl.Utf8).alias("test_subset"),
        pl.col("train.groups").alias("train_source"),
        pl.col("test.fold").alias("fold"),
        pl.col("classif.auc").alias("auc"),
        pl.col("percent.error").alias("percent_error"),
    ).sort(["test_subset", "train_source", "fold"])


def load_reference(path: Path) -> pl.DataFrame:
    """One of our R reference runs, keeping the full splits."""
    frame = pl.read_csv(path, infer_schema_length=20000)
    if "learner_id" in frame.columns:
        frame = frame.filter(pl.col("learner_id") == "classif.cv_glmnet")
    return (
        frame.select(
            pl.col("test.subset").cast(pl.Utf8).alias("test_subset"),
            pl.col("train.subsets").alias("train_source"),
            pl.col("test.fold").alias("fold"),
            pl.col("n.train.groups"),
            pl.col("classif.auc").alias("auc"),
            pl.col("classif.acc").alias("accuracy"),
        )
        .sort("n.train.groups", descending=True)
        .unique(subset=["test_subset", "train_source", "fold"], keep="first")
        .sort(["test_subset", "train_source", "fold"])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()

    registry_path, reference_path = Path(args.registry), Path(args.reference)
    for label, path in (("registry", registry_path), ("reference", reference_path)):
        if not path.is_file():
            print(f"FAIL: {label} not found at {path}", file=sys.stderr)
            return 2

    registry = load_registry(registry_path)
    reference = load_reference(reference_path)
    print(f"registry  {registry.height} rows from {registry_path.name}")
    print(f"reference {reference.height} rows from {reference_path.name}")

    test_subsets = sorted(registry["test_subset"].unique().to_list())

    # ------------------------------------------------------- aggregate
    print("\n" + "=" * 84)
    print("Mean AUC per cell, which is invariant to how folds are numbered")
    print("=" * 84)
    print(f"{'subset':>8}{'source':>8}{'registry':>12}{'reference':>12}{'diff':>11}{'paper':>10}")
    aggregate_gaps = []
    for test_subset in test_subsets:
        for train_source in TRAIN_SOURCES:
            registry_cell = registry.filter(
                (pl.col("test_subset") == test_subset) & (pl.col("train_source") == train_source)
            )["auc"].to_numpy()
            reference_cell = reference.filter(
                (pl.col("test_subset") == test_subset) & (pl.col("train_source") == train_source)
            )["auc"].to_numpy()
            if len(registry_cell) == 0 or len(reference_cell) == 0:
                print(f"{test_subset:>8}{train_source:>8}{'missing':>12}")
                continue
            registry_mean = float(registry_cell.mean())
            reference_mean = float(reference_cell.mean())
            aggregate_gaps.append(abs(registry_mean - reference_mean))
            published = PAPER_2020.get(train_source) if test_subset == "2020" else None
            published_text = f"{published:.4f}" if published is not None else ""
            print(
                f"{test_subset:>8}{train_source:>8}{registry_mean:>12.6f}"
                f"{reference_mean:>12.6f}{registry_mean - reference_mean:>+11.6f}"
                f"{published_text:>10}"
            )
    print(f"\n  mean absolute difference in cell means: {np.mean(aggregate_gaps):.6f}")

    # -------------------------------------------------------- per fold
    print("\n" + "=" * 84)
    print("Per fold, as numbered against sorted")
    print("=" * 84)
    print(
        "  A high correlation as numbered means the fold assignments line up.\n"
        "  A low one there but a high one sorted means both analyses saw the same\n"
        "  spread of folds under different numbering, so only aggregates compare.\n"
    )
    print(f"{'subset':>8}{'source':>8}{'as numbered':>14}{'sorted':>10}{'mean |diff|':>14}")
    as_numbered = []
    for test_subset in test_subsets:
        for train_source in TRAIN_SOURCES:
            registry_cell = (
                registry.filter(
                    (pl.col("test_subset") == test_subset)
                    & (pl.col("train_source") == train_source)
                )
                .sort("fold")["auc"]
                .to_numpy()
            )
            reference_cell = (
                reference.filter(
                    (pl.col("test_subset") == test_subset)
                    & (pl.col("train_source") == train_source)
                )
                .sort("fold")["auc"]
                .to_numpy()
            )
            if len(registry_cell) != len(reference_cell) or len(registry_cell) < 2:
                continue
            paired = float(np.corrcoef(registry_cell, reference_cell)[0, 1])
            sorted_pair = float(np.corrcoef(np.sort(registry_cell), np.sort(reference_cell))[0, 1])
            as_numbered.append(paired)
            print(
                f"{test_subset:>8}{train_source:>8}{paired:>14.5f}{sorted_pair:>10.5f}"
                f"{float(np.abs(registry_cell - reference_cell).mean()):>14.6f}"
            )

    print("\n" + "=" * 84)
    print("Reading this")
    print("=" * 84)
    mean_paired = float(np.mean(as_numbered)) if as_numbered else float("nan")
    print(f"  mean correlation as numbered: {mean_paired:.4f}")
    if mean_paired > 0.9:
        print("  The fold assignments line up. The registry can serve as a reference")
        print("  per fold, the same way the other runs do.")
    elif mean_paired > 0.5:
        print("  Partial agreement, which is the awkward case. Worth understanding")
        print("  before treating the registry as fold-comparable.")
    else:
        print("  The fold assignments do not line up, which is what the change from")
        print("  same_other_cv to same_other_sizes_cv would predict. Compare cell")
        print("  means only; per-fold pairing would be meaningless.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
