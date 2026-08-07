"""What did R's lasso actually keep?

The saved coefficients are the fitted models, so this is a direct look at
which of the 364 features survive the penalty and how large their weights
are. Three questions in particular:

  1. Is Unique_Household_ID among the retained features? A household
     identifier cannot generalise to new households, so if the lasso keeps
     it that is worth knowing independently of anything about the port.
  2. How sparse is the fit, and how stable is that sparsity across splits?
  3. Which features carry the most weight, and do they look like the
     published Figure 4 predictors?

Signs are flipped on load. glmnet oriented its coefficients toward y = 0,
verified by every split reproducing exactly 1 - AUC before the correction,
so a positive weight here means "raises the probability of the outcome".

Aggregations go through numpy rather than polars' own reducers, whose return
types are a union wide enough that mypy cannot narrow them to float.

Run from the repository root::

    uv run python analyses/inspect_r_coefficients.py \\
        --coefficients /path/to/results/coefficients/NSCH_seed1_coefficients.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

WATCH_FEATURES = (
    "Unique_Household_ID",
    "Selected_Child_Weight",
    "Birth_Month",
    "Birth_Year",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coefficients", required=True)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    path = Path(args.coefficients)
    if not path.is_file():
        print(f"FAIL: {path} not found", file=sys.stderr)
        return 2

    # Negate: glmnet's coefficients point toward y = 0 in this fixture.
    coefficients = pl.read_csv(path).with_columns((-pl.col("weight")).alias("weight"))
    n_splits = coefficients.select(["test_subset", "train_source", "fold"]).unique().height
    n_features = coefficients["feature"].n_unique()
    print(f"{coefficients.height} rows, {n_splits} splits, {n_features} features")

    # ---------------------------------------------------------- sparsity
    per_split = (
        coefficients.filter(pl.col("weight") != 0)
        .group_by(["test_subset", "train_source"])
        .agg(pl.len().alias("nonzero_total"))
        .with_columns((pl.col("nonzero_total") / 10).alias("nonzero_per_split"))
        .sort(["test_subset", "train_source"])
    )
    print("\nnonzero coefficients, averaged over the 10 folds")
    print(f"{'subset':>8}{'source':>8}{'nonzero':>10}")
    for row in per_split.iter_rows(named=True):
        print(f"{row['test_subset']:>8}{row['train_source']:>8}{row['nonzero_per_split']:>10.1f}")

    # -------------------------------------------------- watched features
    print("\nfeatures worth checking specifically")
    print(f"{'feature':<26}{'splits nonzero':>16}{'mean weight':>14}{'max |weight|':>14}")
    for feature in WATCH_FEATURES:
        rows = coefficients.filter(pl.col("feature") == feature)
        if rows.height == 0:
            print(f"{feature:<26}{'not in design':>16}")
            continue
        weights = rows["weight"].to_numpy().astype(np.float64)
        nonzero = weights[weights != 0]
        mean_weight = float(nonzero.mean()) if nonzero.size else 0.0
        max_abs = float(np.abs(weights).max())
        print(
            f"{feature:<26}{nonzero.size:>10} of {rows.height:<3}"
            f"{mean_weight:>14.5f}{max_abs:>14.5f}"
        )

    # -------------------------------------------------------- top weights
    print(f"\ntop {args.top} features by mean absolute weight across all splits")
    top = (
        coefficients.group_by("feature")
        .agg(
            pl.col("weight").abs().mean().alias("mean_abs"),
            pl.col("weight").mean().alias("mean_signed"),
            (pl.col("weight") != 0).sum().alias("n_nonzero"),
        )
        .sort("mean_abs", descending=True)
        .head(args.top)
    )
    print(f"{'feature':<62}{'mean wt':>10}{'nonzero':>9}")
    for row in top.iter_rows(named=True):
        name = row["feature"]
        display = name if len(name) <= 60 else name[:57] + "..."
        print(f"{display:<62}{row['mean_signed']:>10.4f}{row['n_nonzero']:>6} /{n_splits:>3}")

    # ------------------------------------------- always-selected features
    selection_counts = coefficients.group_by("feature").agg(
        (pl.col("weight") != 0).sum().alias("n_nonzero")
    )
    n_always = selection_counts.filter(pl.col("n_nonzero") == n_splits).height
    n_never = selection_counts.filter(pl.col("n_nonzero") == 0).height
    print(f"\nfeatures selected in all {n_splits} splits: {n_always}")
    print(f"features never selected in any split: {n_never}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
