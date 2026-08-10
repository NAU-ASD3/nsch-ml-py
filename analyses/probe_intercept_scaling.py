"""Does liblinear's penalized intercept matter for this comparison?

glmnet does not penalize the intercept. Neither does scikit-learn's `saga`.
`liblinear` does: it appends a constant column to the design and penalizes
its coefficient like any other, which biases the intercept toward zero and
shifts the selected penalty to compensate.

`intercept_scaling` controls the magnitude of that appended column. The
penalty on the intercept falls as the scaling rises, so a large value
approximates an unpenalized intercept. The default is 1, which is the
strongly-penalized end.

This fits one split at a fixed C across several scalings and reports what
moves: the intercept itself, the feature coefficients, the held-out
probabilities, and the AUC. It also fits the same split with `saga`, which
does not penalize the intercept, as the reference point.

If the coefficients and probabilities are flat across scalings, the question
is closed and the comparison can proceed on liblinear. If they move, the
60-split run needs rerunning with a scaling that matches glmnet's behaviour,
and the finding belongs in docs/design-decisions.md.

Run from the repository root::

    uv run python analyses/probe_intercept_scaling.py

Fits one split several times, so a couple of minutes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from nsch_ml.soak import assign_folds, iter_soak_splits

DEFAULT_FIXTURE = Path.home() / "Documents/NAU/Grad/Research/ADSI/soak_fixture"
N_FOLDS = 10
SEED = 1
# The C selected by liblinear on the first split in the timing probe. Held
# fixed so the only thing varying is the intercept treatment.
FIXED_C = 0.06021
SCALINGS = (1.0, 10.0, 100.0, 1000.0)


def fit_once(
    x_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.int64],
    x_test: npt.NDArray[np.float64],
    solver: str,
    intercept_scaling: float,
) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the intercept, the coefficients, and the held-out probabilities."""
    kwargs: dict[str, object] = {"C": FIXED_C, "max_iter": 5000}
    if solver == "liblinear":
        kwargs |= {
            "solver": "liblinear",
            "penalty": "l1",
            "intercept_scaling": intercept_scaling,
        }
    else:
        kwargs |= {"solver": "saga", "penalty": "elasticnet", "l1_ratio": 1.0}
    model = LogisticRegression(**kwargs)
    model.fit(x_train, y_train)
    return (
        float(model.intercept_[0]),
        np.asarray(model.coef_[0], dtype=np.float64),
        np.asarray(model.predict_proba(x_test)[:, 1], dtype=np.float64),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    args = parser.parse_args()

    root = Path(os.environ.get("NSCH_SOAK_FIXTURE", args.fixture))
    design_path = root / "data_Classif" / "NSCH_autism.csv"
    folds_path = root / "nsch_autism_folds.csv"
    if not design_path.is_file() or not folds_path.is_file():
        print(f"FAIL: fixture not found under {root}", file=sys.stderr)
        return 2

    design = pl.read_csv(design_path)
    folds = pl.read_csv(folds_path)
    feature_names = [name for name in design.columns if name not in ("survey_year", "y")]
    x_all = design.select(feature_names).to_numpy().astype(np.float64)
    y_all = (design["y"].to_numpy().astype(str) == "Yes").astype(np.int64)
    subset = folds["test.subset"].cast(pl.Utf8).to_numpy()

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
    split = splits[0]
    print(
        f"split: {split.test_subset} {split.train_source.value} fold {split.fold}, "
        f"{len(split.train_idx)} train, {len(split.test_idx)} test"
    )
    print(f"fixed C: {FIXED_C}\n")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_all[split.train_idx])
    x_test = scaler.transform(x_all[split.test_idx])
    y_train, y_test = y_all[split.train_idx], y_all[split.test_idx]

    print("=" * 78)
    print("liblinear across intercept_scaling, and saga for reference")
    print("=" * 78)
    print(f"{'solver':>12}{'scaling':>10}{'intercept':>12}{'nonzero':>9}{'AUC':>11}")

    results: dict[str, tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
    for scaling in SCALINGS:
        intercept, coefficients, probabilities = fit_once(
            x_train, y_train, x_test, "liblinear", scaling
        )
        results[f"liblinear@{scaling:g}"] = (intercept, coefficients, probabilities)
        print(
            f"{'liblinear':>12}{scaling:>10g}{intercept:>12.5f}"
            f"{int((coefficients != 0).sum()):>9}"
            f"{roc_auc_score(y_test, probabilities):>11.6f}"
        )

    intercept, coefficients, probabilities = fit_once(x_train, y_train, x_test, "saga", 1.0)
    results["saga"] = (intercept, coefficients, probabilities)
    print(
        f"{'saga':>12}{'n/a':>10}{intercept:>12.5f}"
        f"{int((coefficients != 0).sum()):>9}"
        f"{roc_auc_score(y_test, probabilities):>11.6f}"
    )

    # -------------------------------------------------- against saga
    print("\n" + "=" * 78)
    print("Each liblinear fit against saga, which leaves the intercept unpenalized")
    print("=" * 78)
    saga_intercept, saga_coefficients, saga_probabilities = results["saga"]
    print(
        f"{'fit':>16}{'intercept gap':>15}{'coef mean |d|':>15}"
        f"{'prob mean |d|':>15}{'Spearman':>11}"
    )
    for name, (intercept, coefficients, probabilities) in results.items():
        if name == "saga":
            continue
        rho = float(stats.spearmanr(probabilities, saga_probabilities).statistic)
        print(
            f"{name:>16}{abs(intercept - saga_intercept):>15.5f}"
            f"{float(np.abs(coefficients - saga_coefficients).mean()):>15.6f}"
            f"{float(np.abs(probabilities - saga_probabilities).mean()):>15.6f}"
            f"{rho:>11.5f}"
        )

    # -------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print("Reading this")
    print("=" * 78)
    default_intercept = results["liblinear@1"][0]
    largest_intercept = results[f"liblinear@{SCALINGS[-1]:g}"][0]
    drift = abs(largest_intercept - default_intercept)
    print(f"  intercept at scaling 1      : {default_intercept:+.5f}")
    print(f"  intercept at scaling {SCALINGS[-1]:g}   : {largest_intercept:+.5f}")
    print(f"  intercept moved by          : {drift:.5f}")
    print(f"  saga intercept              : {saga_intercept:+.5f}")
    print()
    print("  If the intercept barely moves across scalings and liblinear sits close")
    print("  to saga on coefficients and probabilities, the penalized intercept is")
    print("  immaterial here and the comparison can proceed as planned.")
    print()
    print("  If it moves and the default is far from saga, the 60-split run needs")
    print("  redoing with a scaling that approximates an unpenalized intercept,")
    print("  and the choice belongs in docs/design-decisions.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
