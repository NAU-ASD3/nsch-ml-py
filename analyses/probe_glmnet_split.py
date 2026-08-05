"""Timing and AUC probe on a single SOAK split before scaling up.

Fits one split -- test subset 2019, train source ALL, fold 1, not
downsampled -- and compares the test AUC against the R reference from
``reproduce-soak-nsch/results/2026-03-06/NSCH_proj.csv``, where
``classif.cv_glmnet`` scored 0.968567898692473 on the same split.

The point is to learn two things before committing to 60 fits:
how long one fit takes, and whether the AUC lands in the right
neighbourhood. Exact agreement is not expected. ``cv_glmnet`` defaults
to lasso with the ``lambda.1se`` rule; scikit-learn picks its penalty
strength differently. The appendix reports ridge at 0.9685841 and
lasso at 0.9682405 on a related fit, so the signal is strong enough
that alpha barely moves the AUC. Landing within ~0.005 is the bar.

Run from the repository root::

    uv run python analyses/probe_glmnet_split.py

Add ``--drop-hhid`` to exclude ``Unique_Household_ID``. NSCH household
IDs are year-prefixed, so leaving it in lets a model read the survey
year directly off a "feature" -- which matters precisely because SAME,
OTHER, and ALL differ in which years they train on.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from nsch_ml.soak import TrainSource, assign_folds, iter_soak_splits

DEFAULT_FIXTURE = Path.home() / "Documents/NAU/Grad/Research/ADSI/soak_fixture"
N_FOLDS = 10
SEED = 1
EXPECTED_ROWS = 46010

# reproduce-soak-nsch/results/2026-03-06/NSCH_proj.csv, iteration 1.
R_AUC = 0.968567898692473
R_ACC = 0.979132344865459
TARGET_SUBSET = "2019"
TARGET_FOLD = 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-hhid", action="store_true")
    ap.add_argument("--cv", action="store_true", help="also time LogisticRegressionCV")
    args = ap.parse_args()

    root = Path(os.environ.get("NSCH_SOAK_FIXTURE", DEFAULT_FIXTURE))
    print(f"fixture: {root}")
    if not root.is_dir():
        print("FAIL: fixture directory not found", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    folds = pl.read_csv(root / "nsch_autism_folds.csv")
    design = pl.read_csv(root / "data_Classif" / "NSCH_autism.csv")
    print(f"load: {time.perf_counter() - t0:.1f}s   design {design.shape}")
    if design.height != EXPECTED_ROWS or folds.height != EXPECTED_ROWS:
        print("FAIL: unexpected row count", file=sys.stderr)
        return 2

    drop = ["survey_year", "y"]
    if args.drop_hhid:
        drop.append("Unique_Household_ID")
    feature_cols = [c for c in design.columns if c not in drop]
    print(f"features: {len(feature_cols)}   hhid dropped: {args.drop_hhid}")

    x_all = design.select(feature_cols).to_numpy().astype(np.float64)
    y_all = (design["y"].to_numpy().astype(str) == "Yes").astype(np.int64)
    subset = folds["test.subset"].cast(pl.Utf8).to_numpy()
    print(f"positives: {int(y_all.sum())} / {len(y_all)} ({y_all.mean():.4%})")

    fold_ids = assign_folds(
        subset=subset,
        outcome=design["y"],
        n_folds=N_FOLDS,
        precomputed=folds["fold"],
    )
    chosen = None
    for split in iter_soak_splits(
        fold_ids=fold_ids,
        subset=subset,
        outcome=design["y"],
        sizes=0,
        seed=SEED,
    ):
        if (
            split.test_subset == TARGET_SUBSET
            and split.fold == TARGET_FOLD
            and split.train_source is TrainSource.ALL
            and not split.downsampled
        ):
            chosen = split
            break
    if chosen is None:
        print("FAIL: target split not produced", file=sys.stderr)
        return 2
    print(f"split: train {len(chosen.train_idx)}   test {len(chosen.test_idx)}")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_all[chosen.train_idx])
    x_test = scaler.transform(x_all[chosen.test_idx])
    y_train, y_test = y_all[chosen.train_idx], y_all[chosen.test_idx]

    print("\n== single fit, L2, C=1 ==")
    t0 = time.perf_counter()
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000)
    clf.fit(x_train, y_train)
    elapsed = time.perf_counter() - t0
    prob = clf.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    acc = accuracy_score(y_test, (prob >= 0.5).astype(np.int64))
    print(f"fit time: {elapsed:.1f}s")
    print(f"AUC {auc:.7f}   R {R_AUC:.7f}   delta {auc - R_AUC:+.7f}")
    print(f"ACC {acc:.7f}   R {R_ACC:.7f}   delta {acc - R_ACC:+.7f}")
    print(f"projected for 60 full splits: {elapsed * 60 / 60:.1f} min")

    if args.cv:
        print("\n== LogisticRegressionCV, 10 Cs, 5-fold inner ==")
        t0 = time.perf_counter()
        cv = LogisticRegressionCV(
            Cs=10,
            cv=5,
            penalty="l2",
            solver="lbfgs",
            scoring="roc_auc",
            max_iter=2000,
            n_jobs=-1,
        )
        cv.fit(x_train, y_train)
        elapsed_cv = time.perf_counter() - t0
        prob = cv.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, prob)
        print(f"fit time: {elapsed_cv:.1f}s   chosen C: {cv.C_[0]:.5g}")
        print(f"AUC {auc:.7f}   R {R_AUC:.7f}   delta {auc - R_AUC:+.7f}")
        print(f"projected for 60 full splits: {elapsed_cv * 60 / 60:.1f} min")

    print("\n== note ==")
    print("Exact agreement is not expected and not the bar. |dAUC| < 0.005")
    print("means the pipeline reproduces the published discrimination.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
