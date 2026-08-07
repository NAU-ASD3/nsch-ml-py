"""How much does the R reference disagree with itself across seeds?

Three R runs exist that differ only in the `cv.glmnet` inner cross-validation
seed: same machine, same day, same package build, same outer folds. Anything
they disagree about is irreducible noise in the reference, and no
reimplementation can be expected to sit closer to one of them than they sit
to each other.

That makes this the natural yardstick for an equivalence margin. It is
computed here before any Python comparison is run, so the margin it anchors
cannot be tuned toward a result we have already seen.

Three quantities, matching what the Python comparison will use:

  Coefficients      mean and max absolute difference over the 364 weights,
                    plus how often the two runs disagree about which
                    features are selected at all.
  Score rankings    Spearman correlation of the held-out linear predictors,
                    on 1,819 to 2,781 children per split.
  AUC               the fold-level summary, for continuity with earlier work.

Note that only 30 of the 60 splits hold distinct models: for a given fold,
"same" on 2019 and "other" on 2020 train on identical rows. Coefficient
statistics are therefore computed over the distinct fits.

Signs are left as glmnet wrote them. Differences are unaffected, and both
sides carry the same convention.

Run from the repository root::

    uv run python analyses/r_seed_yardstick.py \\
        --coefficient-dir /path/to/reproduce-soak-nsch/results/coefficients
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

DEFAULT_FIXTURE = Path.home() / "Documents/NAU/Grad/Research/ADSI/soak_fixture"
SPLIT_KEY = ["test_subset", "train_source", "fold"]
# For a given fold these two cells train on the same rows, so keeping one of
# each pair leaves the distinct fitted models.
DISTINCT_CELLS = (("2019", "same"), ("2019", "other"), ("2019", "all"))


def load_coefficients(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(path).with_columns(pl.col("test_subset").cast(pl.Utf8))
    return frame.sort([*SPLIT_KEY, "feature"])


def coefficient_comparison(left: pl.DataFrame, right: pl.DataFrame) -> dict[str, float]:
    """Weight differences over the distinct fitted models."""
    keep = pl.any_horizontal(
        [
            (pl.col("test_subset") == subset) & (pl.col("train_source") == source)
            for subset, source in DISTINCT_CELLS
        ]
    )
    left_weights = left.filter(keep)["weight"].to_numpy()
    right_weights = right.filter(keep)["weight"].to_numpy()
    if left_weights.shape != right_weights.shape:
        raise ValueError(f"shape mismatch: {left_weights.shape} against {right_weights.shape}")
    differences = np.abs(left_weights - right_weights)
    left_selected = left_weights != 0
    right_selected = right_weights != 0
    return {
        "mean_abs": float(differences.mean()),
        "max_abs": float(differences.max()),
        "selection_disagreement": float((left_selected != right_selected).mean()),
        "n_weights": float(len(differences)),
    }


def ranking_comparison(
    left: pl.DataFrame,
    right: pl.DataFrame,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    subset_of_row: np.ndarray,
    fold_of_row: np.ndarray,
) -> dict[str, float]:
    """Spearman correlation of held-out linear predictors, per split."""
    correlations = []
    keys = left.select(SPLIT_KEY).unique().sort(SPLIT_KEY)
    for key in keys.iter_rows(named=True):
        selector = (
            (pl.col("test_subset") == key["test_subset"])
            & (pl.col("train_source") == key["train_source"])
            & (pl.col("fold") == key["fold"])
        )
        left_cell = left.filter(selector)
        right_cell = right.filter(selector)
        left_lookup = dict(zip(left_cell["feature"], left_cell["weight"], strict=True))
        right_lookup = dict(zip(right_cell["feature"], right_cell["weight"], strict=True))
        left_beta = np.array([left_lookup[name] for name in feature_names])
        right_beta = np.array([right_lookup[name] for name in feature_names])

        is_test_row = (subset_of_row == key["test_subset"]) & (fold_of_row == key["fold"])
        test_rows = feature_matrix[is_test_row]
        correlations.append(
            float(stats.spearmanr(test_rows @ left_beta, test_rows @ right_beta).statistic)
        )
    values = np.array(correlations)
    return {"min": float(values.min()), "mean": float(values.mean()), "n": float(len(values))}


def auc_comparison(left: pl.DataFrame, right: pl.DataFrame) -> dict[str, float]:
    left_auc = left.group_by(SPLIT_KEY).agg(pl.col("r_auc").first()).sort(SPLIT_KEY)
    right_auc = right.group_by(SPLIT_KEY).agg(pl.col("r_auc").first()).sort(SPLIT_KEY)
    differences = np.abs(left_auc["r_auc"].to_numpy() - right_auc["r_auc"].to_numpy())
    return {"mean_abs": float(differences.mean()), "max_abs": float(differences.max())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coefficient-dir", required=True)
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--runs", nargs="+", default=["seed1", "seed2", "seed3"])
    args = parser.parse_args()

    coefficient_dir = Path(args.coefficient_dir)
    runs: dict[str, pl.DataFrame] = {}
    for name in args.runs:
        path = coefficient_dir / f"NSCH_{name}_coefficients.csv"
        if not path.is_file():
            print(f"FAIL: {path} not found", file=sys.stderr)
            return 2
        runs[name] = load_coefficients(path)
        print(f"{name}: {runs[name].height} rows")
    if len(runs) < 2:
        print("need at least two runs", file=sys.stderr)
        return 1

    fixture = Path(os.environ.get("NSCH_SOAK_FIXTURE", args.fixture))
    design = pl.read_csv(fixture / "data_Classif" / "NSCH_autism.csv")
    folds = pl.read_csv(fixture / "nsch_autism_folds.csv")
    feature_names = runs[args.runs[0]].head(364)["feature"].to_list()
    feature_matrix = design.select(feature_names).to_numpy().astype(np.float64)
    subset_of_row = folds["test.subset"].cast(pl.Utf8).to_numpy()
    fold_of_row = folds["fold"].to_numpy()
    print(f"design {feature_matrix.shape}, {len(feature_names)} features\n")

    print("=" * 78)
    print("Coefficients, over the 30 distinct fitted models")
    print("=" * 78)
    print(f"{'pair':<20}{'mean |diff|':>14}{'max |diff|':>13}{'selection differs':>19}")
    coefficient_means = []
    for left_name, right_name in itertools.combinations(args.runs, 2):
        result = coefficient_comparison(runs[left_name], runs[right_name])
        coefficient_means.append(result["mean_abs"])
        print(
            f"{left_name + ' vs ' + right_name:<20}{result['mean_abs']:>14.6f}"
            f"{result['max_abs']:>13.6f}{result['selection_disagreement']:>18.1%}"
        )

    print("\n" + "=" * 78)
    print("Held-out score rankings, Spearman per split")
    print("=" * 78)
    print(f"{'pair':<20}{'min rho':>12}{'mean rho':>12}{'splits':>9}")
    ranking_minima = []
    for left_name, right_name in itertools.combinations(args.runs, 2):
        result = ranking_comparison(
            runs[left_name],
            runs[right_name],
            feature_matrix,
            feature_names,
            subset_of_row,
            fold_of_row,
        )
        ranking_minima.append(result["min"])
        print(
            f"{left_name + ' vs ' + right_name:<20}{result['min']:>12.5f}"
            f"{result['mean']:>12.5f}{int(result['n']):>9}"
        )

    print("\n" + "=" * 78)
    print("Fold-level AUC, for continuity with earlier work")
    print("=" * 78)
    print(f"{'pair':<20}{'mean |diff|':>14}{'max |diff|':>13}")
    auc_means = []
    for left_name, right_name in itertools.combinations(args.runs, 2):
        result = auc_comparison(runs[left_name], runs[right_name])
        auc_means.append(result["mean_abs"])
        print(
            f"{left_name + ' vs ' + right_name:<20}{result['mean_abs']:>14.6f}"
            f"{result['max_abs']:>13.6f}"
        )

    print("\n" + "=" * 78)
    print("Yardstick")
    print("=" * 78)
    print(f"  coefficient mean |diff|, worst pair : {max(coefficient_means):.6f}")
    print(f"  ranking Spearman, worst split       : {min(ranking_minima):.5f}")
    print(f"  AUC mean |diff|, worst pair         : {max(auc_means):.6f}")
    print()
    print("  These are the reference disagreeing with itself when only the inner")
    print("  cross-validation seed changes. A reimplementation cannot reasonably")
    print("  be held to a tighter standard, and a margin set from these numbers")
    print("  is anchored to the analysis rather than chosen by hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
