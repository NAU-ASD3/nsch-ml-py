"""Does the pipeline reproduce the published featureless baseline?

A featureless classifier ignores every feature and predicts the training
set's class proportion for every held-out child. It is the floor the other
learners are measured against, and the SOAK paper ran it on this task
alongside `cv_glmnet`.

It is also the cleanest end-to-end check available on everything except the
model. Its predictions depend on exactly two things: which children are in
the training set, and how many of them have the outcome. Nothing about
penalties, solvers, or convergence enters. If our numbers match the published
ones, then the data, the outcome coding, and the train and test membership
are all right, and any later disagreement on a real learner belongs to the
learner.

Two consequences of predicting a constant.

AUC is 0.5 by construction, because every child gets the same score and no
ordering exists. scikit-learn returns 0.5 for that case. It carries no
information here and is reported only to confirm both sides agree it is
degenerate.

Accuracy is what matters, and it depends only on class balance. At a 3% base
rate the constant prediction sits below 0.5, so every child is classified as
not having the outcome, and accuracy equals the share of held-out children
without it. That quantity is a property of the data, not of how the folds
were drawn, which is what makes this comparable to the published run even
though its fold assignment differs from ours.

Run from the repository root::

    uv run python analyses/run_featureless_replication.py \\
        --registry $PAPER/data_Classif_batchmark_registry.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, roc_auc_score

from nsch_ml.soak import assign_folds, iter_soak_splits

DEFAULT_FIXTURE = Path.home() / "Documents/NAU/Grad/Research/ADSI/soak_fixture"
N_FOLDS = 10
SEED = 1
EXPECTED_ROWS = 46010
EXPECTED_SPLITS = 60
TRAIN_SOURCES = ("same", "other", "all")
# The published featureless run and ours should agree on accuracy to about
# this much. Both are the share of held-out children without the outcome,
# differing only in which children each fold happens to hold.
ACCURACY_TOLERANCE = 0.005


def load_published(path: Path) -> pl.DataFrame | None:
    """Cell averages for the published featureless run, if it is available."""
    if not path.is_file():
        return None
    rows = pl.read_csv(path, infer_schema_length=20000).filter(
        (pl.col("task_id") == "NSCH_autism") & (pl.col("algorithm") == "featureless")
    )
    if rows.height == 0:
        print(f"  no featureless rows for NSCH_autism in {path.name}")
        return None
    return (
        rows.select(
            pl.col("test.group").cast(pl.Utf8).alias("test_subset"),
            pl.col("train.groups").alias("train_source"),
            pl.col("classif.auc").alias("auc"),
            (1.0 - pl.col("percent.error") / 100.0).alias("accuracy"),
        )
        .group_by(["test_subset", "train_source"])
        .agg(
            pl.col("auc").mean().alias("published_auc"),
            pl.col("accuracy").mean().alias("published_accuracy"),
        )
        .sort(["test_subset", "train_source"])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--registry", default=None, help="the published results CSV")
    parser.add_argument("--out", default="analyses/featureless_replication.csv")
    args = parser.parse_args()

    root = Path(os.environ.get("NSCH_SOAK_FIXTURE", args.fixture))
    design_path = root / "data_Classif" / "NSCH_autism.csv"
    folds_path = root / "nsch_autism_folds.csv"
    if not design_path.is_file() or not folds_path.is_file():
        print(f"FAIL: fixture not found under {root}", file=sys.stderr)
        return 2

    design = pl.read_csv(design_path)
    folds = pl.read_csv(folds_path)
    if design.height != EXPECTED_ROWS:
        print(f"FAIL: {design.height} rows, expected {EXPECTED_ROWS}", file=sys.stderr)
        return 2

    outcome = (design["y"].to_numpy().astype(str) == "Yes").astype(np.int64)
    subset = folds["test.subset"].cast(pl.Utf8).to_numpy()
    print(f"design {design.shape}, overall outcome rate {outcome.mean():.4f}")

    fold_ids = assign_folds(
        subset=subset, outcome=design["y"], n_folds=N_FOLDS, precomputed=folds["fold"]
    )
    splits = [
        split
        for split in iter_soak_splits(
            fold_ids=fold_ids, subset=subset, outcome=design["y"], sizes=0, seed=SEED
        )
        if not split.downsampled
    ]
    if len(splits) != EXPECTED_SPLITS:
        print(f"FAIL: {len(splits)} full splits, expected {EXPECTED_SPLITS}", file=sys.stderr)
        return 1
    print(f"fitting {len(splits)} splits\n")

    rows = []
    for split in splits:
        train_outcome = outcome[split.train_idx]
        test_outcome = outcome[split.test_idx]
        # The whole model: the training set's outcome rate, for everyone.
        predicted = float(train_outcome.mean())
        probabilities = np.full(len(test_outcome), predicted)
        rows.append(
            {
                "test_subset": split.test_subset,
                "train_source": split.train_source.value,
                "fold": split.fold,
                "n_train": len(split.train_idx),
                "n_test": len(split.test_idx),
                "train_rate": predicted,
                "test_rate": float(test_outcome.mean()),
                "auc": float(roc_auc_score(test_outcome, probabilities)),
                "accuracy": float(
                    accuracy_score(test_outcome, (probabilities >= 0.5).astype(np.int64))
                ),
            }
        )

    results = pl.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.write_csv(out_path)
    print(f"wrote {out_path}")

    degenerate = results.filter(pl.col("auc") != 0.5)
    if degenerate.height:
        print(
            f"\nWARNING: {degenerate.height} splits have AUC away from 0.5, so the "
            "predictions are not constant. Something is wrong."
        )

    cells = (
        results.group_by(["test_subset", "train_source"])
        .agg(
            pl.col("auc").mean().alias("our_auc"),
            pl.col("accuracy").mean().alias("our_accuracy"),
            pl.col("test_rate").mean().alias("mean_test_rate"),
        )
        .sort(["test_subset", "train_source"])
    )

    print("\n" + "=" * 78)
    print("Our featureless baseline, averaged over the 10 folds in each cell")
    print("=" * 78)
    print(f"{'subset':>8}{'source':>8}{'AUC':>9}{'accuracy':>11}{'outcome rate':>15}")
    for row in cells.iter_rows(named=True):
        print(
            f"{row['test_subset']:>8}{row['train_source']:>8}{row['our_auc']:>9.4f}"
            f"{row['our_accuracy']:>11.6f}{row['mean_test_rate']:>15.6f}"
        )
    print("\n  Accuracy should equal one minus the outcome rate, since every child")
    print("  is classified as not having the outcome.")

    if not args.registry:
        print("\n(no --registry given, so no comparison against the published run)")
        return 0

    published = load_published(Path(args.registry))
    if published is None:
        print(f"\nFAIL: could not read published featureless rows from {args.registry}")
        return 1

    merged = cells.join(published, on=["test_subset", "train_source"], how="inner")
    if merged.height != cells.height:
        print(
            f"\nFAIL: matched {merged.height} of {cells.height} cells against the published run",
            file=sys.stderr,
        )
        return 1

    print("\n" + "=" * 78)
    print("Against the published featureless run")
    print("=" * 78)
    print(f"{'subset':>8}{'source':>8}{'ours':>12}{'published':>12}{'diff':>11}   ok")
    worst = 0.0
    n_failed = 0
    for row in merged.iter_rows(named=True):
        difference = row["our_accuracy"] - row["published_accuracy"]
        worst = max(worst, abs(difference))
        passed = abs(difference) < ACCURACY_TOLERANCE
        n_failed += int(not passed)
        print(
            f"{row['test_subset']:>8}{row['train_source']:>8}"
            f"{row['our_accuracy']:>12.6f}{row['published_accuracy']:>12.6f}"
            f"{difference:>+11.6f}   {'yes' if passed else 'NO'}"
        )

    print(f"\nworst absolute difference in accuracy: {worst:.6f}")
    if n_failed:
        print(f"FAIL: {n_failed} of {merged.height} cells outside {ACCURACY_TOLERANCE}")
        print("The data or the split membership differs from the published run.")
        return 1
    print(f"PASS: all {merged.height} cells agree within {ACCURACY_TOLERANCE}.")
    print("The data, the outcome coding, and the split membership all reproduce")
    print("the published run on a model with no moving parts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
