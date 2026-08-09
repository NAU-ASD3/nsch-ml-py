"""Repair R's prediction export, and check the repair against R's own AUC.

The R run wrote two problems into `NSCH_seed1_predictions.csv`, both from
class-label conventions rather than from anything about the fit:

  1. mlr3 took "No" as the positive class, so the saved probability is
     P(no autism services). We want P(outcome), which is one minus it. This
     is the same convention that made glmnet's coefficients point toward the
     negative class, and like that one it leaves AUC untouched while
     inverting everything on the probability scale.

  2. `truth` was coerced from a "Yes"/"No" factor with as.integer, which
     produced NA for every row. The outcome is in the design matrix, so it
     is restored by joining on row_id rather than rerunning anything.

Three label conventions are in play across this pipeline and none of them is
visible in an AUC:

  - the design's outcome column `y` holds the strings "Yes" and "No", not 0
    and 1;
  - mlr3 assigned "No" as the positive class;
  - scikit-learn's roc_auc_score, given string labels, treats the
    lexicographically last level as positive, which is "Yes".

They happen to compose correctly once the probability is inverted, but each
one would silently corrupt a probability-scale comparison on its own.

The repaired file is checked two ways before it is written. The mean
predicted probability must land near the outcome prevalence rather than near
its complement, and the recomputed AUC must match R's reported AUC on all 60
splits.

Run from the repository root::

    uv run python analyses/repair_r_predictions.py \\
        --predictions /path/to/results/predictions/NSCH_seed1_predictions.csv \\
        --scores /path/to/results/predictions/NSCH_predictions_scores.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import cast

import polars as pl
from sklearn.metrics import roc_auc_score

DEFAULT_FIXTURE = Path.home() / "Documents/NAU/Grad/Research/ADSI/soak_fixture"
TOLERANCE = 1e-9
EXPECTED_SPLITS = 60
POSITIVE_LABEL = "Yes"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--out", default=None, help="defaults to alongside the input")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    scores_path = Path(args.scores)
    for path in (predictions_path, scores_path):
        if not path.is_file():
            print(f"FAIL: {path} not found", file=sys.stderr)
            return 2

    fixture = Path(os.environ.get("NSCH_SOAK_FIXTURE", args.fixture))
    design = pl.read_csv(fixture / "data_Classif" / "NSCH_autism.csv")
    predictions = pl.read_csv(predictions_path)
    scores = pl.read_csv(scores_path)
    print(f"predictions {predictions.shape}, scores {scores.shape}, design {design.shape}")

    observed_labels = sorted(design["y"].unique().to_list())
    if POSITIVE_LABEL not in observed_labels:
        print(
            f"FAIL: {POSITIVE_LABEL!r} not among the outcome labels {observed_labels}",
            file=sys.stderr,
        )
        return 2
    print(f"outcome labels {observed_labels}, taking {POSITIVE_LABEL!r} as positive")

    # row_id is 1-based from R and indexes the design in its original order,
    # confirmed by its range covering exactly the design height.
    outcome_by_row = design.with_row_index("row_id", offset=1).select(
        "row_id", (pl.col("y") == POSITIVE_LABEL).cast(pl.Int8).alias("truth")
    )
    repaired = (
        predictions.drop("truth")
        .join(outcome_by_row, on="row_id", how="left")
        .with_columns((1.0 - pl.col("r_prob")).alias("r_prob"))
        .with_columns(pl.col("test_subset").cast(pl.Utf8))
    )
    n_missing = int(repaired["truth"].is_null().sum())
    if n_missing:
        print(f"FAIL: {n_missing} rows did not match a design row by row_id", file=sys.stderr)
        return 1
    print(f"restored truth for {repaired.height} rows, inverted the probability")

    # Both columns are numeric by construction, truth from the cast above and
    # r_prob from arithmetic, so cast rather than checking or materialising.
    prevalence = cast("float", repaired["truth"].mean())
    mean_probability = cast("float", repaired["r_prob"].mean())
    print(f"outcome prevalence {prevalence:.4f}, mean predicted probability {mean_probability:.4f}")
    if abs(mean_probability - prevalence) > 0.02:
        print(
            "FAIL: the mean prediction is far from the prevalence, so the class "
            "convention is still wrong",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------- AUC gate
    reported = scores.select(
        pl.col("test.subset").cast(pl.Utf8).alias("test_subset"),
        pl.col("train.subsets").alias("train_source"),
        pl.col("test.fold").alias("fold"),
        pl.col("classif.auc").alias("r_auc"),
    )
    print(f"\nchecking {reported.height} splits against R's reported AUC")
    print(f"{'subset':>8}{'source':>8}{'fold':>6}{'n':>7}{'R AUC':>12}{'rebuilt':>12}{'diff':>11}")

    worst = 0.0
    n_failed = 0
    for row in reported.sort(["test_subset", "train_source", "fold"]).iter_rows(named=True):
        cell = repaired.filter(
            (pl.col("test_subset") == row["test_subset"])
            & (pl.col("train_source") == row["train_source"])
            & (pl.col("fold") == row["fold"])
        )
        rebuilt = float(roc_auc_score(cell["truth"].to_numpy(), cell["r_prob"].to_numpy()))
        difference = abs(rebuilt - row["r_auc"])
        worst = max(worst, difference)
        n_failed += int(difference >= TOLERANCE)
        print(
            f"{row['test_subset']:>8}{row['train_source']:>8}{row['fold']:>6}"
            f"{cell.height:>7}{row['r_auc']:>12.7f}{rebuilt:>12.7f}{difference:>11.2e}"
        )

    print(f"\nworst absolute difference: {worst:.3e}")
    if n_failed:
        print(f"FAIL: {n_failed} of {reported.height} splits outside {TOLERANCE}")
        return 1
    if reported.height != EXPECTED_SPLITS:
        print(f"FAIL: {reported.height} splits, expected {EXPECTED_SPLITS}")
        return 1

    out_path = (
        Path(args.out)
        if args.out
        else predictions_path.with_name(predictions_path.stem + "_repaired.csv")
    )
    repaired.write_csv(out_path)
    print(
        f"PASS: all {EXPECTED_SPLITS} splits reproduce R's AUC from the repaired "
        f"probabilities.\nwrote {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
