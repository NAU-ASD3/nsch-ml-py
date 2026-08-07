"""Do R's saved coefficients reproduce R's reported AUC?

`LearnerClassifCVGlmnetSave` stores the fitted coefficients at the selected
lambda, 364 per split, but not the intercept. That is enough for a great
deal, because the intercept is a single additive constant and AUC is
rank-based: adding a constant to every score changes no ranking, so the AUC
computed from ``X @ beta`` alone must equal the AUC R reported.

That gives a free check on the whole recovery. If the reconstructed AUCs
match R's on all 60 splits, we hold R's fitted models and can compare
coefficients and score rankings directly, without refitting anything. If
they do not match, the reconstruction is wrong and nothing downstream should
be trusted.

Two details that matter. glmnet returns coefficients on the original feature
scale, having standardised internally and back-transformed, so the design
matrix is used raw here with no scaling. And the fixture's column layout is
`survey_year`, `y`, then the 364 features; the subset and fold assignment
both come from the folds file, whose columns are `row_id`, `test.subset`,
`fold`.

Run from the repository root::

    uv run python analyses/verify_r_coefficients.py \\
        --coefficients /path/to/results/coefficients/NSCH_seed1_coefficients.csv

The fixture path comes from NSCH_SOAK_FIXTURE, as elsewhere.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

DEFAULT_FIXTURE = Path.home() / "Documents/NAU/Grad/Research/ADSI/soak_fixture"
# The reconstruction is exact arithmetic. Any disagreement beyond CSV rounding
# means the design, the row selection, or the coefficient scale is wrong.
TOLERANCE = 1e-6
OUTCOME_COLUMN = "y"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coefficients", required=True)
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    args = parser.parse_args()

    fixture = Path(os.environ.get("NSCH_SOAK_FIXTURE", args.fixture))
    design_path = fixture / "data_Classif" / "NSCH_autism.csv"
    folds_path = fixture / "nsch_autism_folds.csv"
    coefficients_path = Path(args.coefficients)
    for path in (design_path, folds_path, coefficients_path):
        if not path.is_file():
            print(f"FAIL: {path} not found", file=sys.stderr)
            return 2

    design = pl.read_csv(design_path)
    folds = pl.read_csv(folds_path)
    coefficients = pl.read_csv(coefficients_path)
    print(f"design {design.shape}, folds {folds.shape}, coefficients {coefficients.shape}")

    if design.height != folds.height:
        print(
            f"FAIL: {design.height} design rows against {folds.height} fold rows", file=sys.stderr
        )
        return 2

    first_split = coefficients.head(364)
    feature_names = first_split["feature"].to_list()
    missing = [name for name in feature_names if name not in design.columns]
    if missing:
        print(
            f"FAIL: {len(missing)} coefficient features absent from the design, "
            f"first few {missing[:5]}",
            file=sys.stderr,
        )
        return 2
    non_features = [column for column in design.columns if column not in set(feature_names)]
    print(
        f"{len(feature_names)} features aligned by name; "
        f"design columns that are not features: {non_features}"
    )

    # Raw, unscaled: glmnet reports coefficients on the original scale.
    feature_matrix = design.select(feature_names).to_numpy().astype(np.float64)
    outcome = design[OUTCOME_COLUMN].to_numpy()
    subset_of_row = folds["test.subset"].cast(pl.Utf8).to_numpy()
    fold_of_row = folds["fold"].to_numpy()

    split_keys = (
        coefficients.select(["test_subset", "train_source", "fold"])
        .unique()
        .sort(["test_subset", "train_source", "fold"])
    )
    print(f"\nchecking {split_keys.height} splits, tolerance {TOLERANCE}\n")
    print(
        f"{'subset':>8}{'source':>8}{'fold':>6}{'n test':>8}"
        f"{'R AUC':>12}{'rebuilt':>12}{'diff':>11}  ok"
    )

    worst_difference = 0.0
    n_failed = 0
    for key in split_keys.iter_rows(named=True):
        cell = coefficients.filter(
            (pl.col("test_subset") == key["test_subset"])
            & (pl.col("train_source") == key["train_source"])
            & (pl.col("fold") == key["fold"])
        )
        weight_of_feature = dict(zip(cell["feature"], cell["weight"], strict=True))
        weights = np.array([weight_of_feature[name] for name in feature_names], dtype=np.float64)

        is_test_row = (subset_of_row == str(key["test_subset"])) & (fold_of_row == key["fold"])
        # glmnet orients coefficients toward the first factor level, which is y = 0
        # here, so the linear predictor scores the negative class. Negating gives
        # scores for y = 1. Verified by every split reproducing exactly 1 - AUC.
        linear_predictor = -(feature_matrix[is_test_row] @ weights)
        rebuilt_auc = float(roc_auc_score(outcome[is_test_row], linear_predictor))
        reported_auc = float(cell["r_auc"][0])
        difference = abs(rebuilt_auc - reported_auc)
        worst_difference = max(worst_difference, difference)
        passed = difference < TOLERANCE
        n_failed += int(not passed)
        print(
            f"{key['test_subset']:>8}{key['train_source']:>8}{key['fold']:>6}"
            f"{int(is_test_row.sum()):>8}{reported_auc:>12.7f}{rebuilt_auc:>12.7f}"
            f"{difference:>11.2e}  {'yes' if passed else 'NO'}"
        )

    print(f"\nworst absolute difference: {worst_difference:.3e}")
    if n_failed:
        print(f"FAIL: {n_failed} of {split_keys.height} splits outside tolerance.")
        print("Do not build on these coefficients until this is understood.")
        return 1
    print(f"PASS: all {split_keys.height} splits reproduce R's AUC.")
    print("R's fitted models are recoverable from the saved coefficients, so")
    print("coefficient and score-ranking comparisons need no further R runs.")
    print("Probability-scale comparison still needs the intercept, which is not stored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
