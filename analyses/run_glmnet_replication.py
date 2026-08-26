"""Replicate the cv_glmnet SOAK results in Python and compare to R.

Fits a penalized logistic regression on every SOAK split and compares
test AUC and accuracy against the R reference in
``reproduce-soak-nsch/results/seed-variation/NSCH_seed1.csv``
(``classif.cv_glmnet``, same fixture, same folds).

Which R run is the reference matters more than it looks. Two R runs that
differ only in a random seed disagree by as much as 0.0206 AUC on
individual splits, an order of magnitude larger than any R-to-Python gap
this script reports. ``NSCH_seed1.csv`` is the run that shares our fold
assignment and was verified row by row, and it is what every committed
result and every figure in ``docs/replication-equivalence.md`` was
produced against. This file previously defaulted to
``results/2026-03-06/NSCH_proj.csv``, a different draw, so a rerun with
the old default produced ``r_auc`` values disagreeing with the committed
results for no reason a reader could see. Override with
``NSCH_SOAK_REFERENCE`` to compare against a different run deliberately.

Why two selection rules
-----------------------
mlr3's ``classif.cv_glmnet`` predicts at ``s = "lambda.1se"`` by
default: the largest penalty whose cross-validated loss is within one
standard error of the minimum. That is deliberately more regularized
than the CV optimum. A single-split probe found scikit-learn's optimum
scoring ~0.005 AUC *above* R, which looked like that rule's signature.
The full run refuted it -- there is no systematic offset, and the probe
simply caught two folds pulling opposite ways. Both rules are still
computed from the same fitted score grid so the question stays
measured rather than assumed.

``LogisticRegressionCV`` retains the per-fold score for every C, so one
fit yields both rules; only the final refit is repeated.

Joining to the R reference
--------------------------
R's ``n.train.groups`` records the *nominal* train size, not the actual
one. For 2019/all/fold 1 R reports 41,408 while the realised train set
holds 41,410 rows -- the same nominal-versus-actual gap that produced
the downsample defect fixed in PR#9. The reference file covers the 60
non-downsampled splits only, so ``(test.subset, train.subsets,
test.fold)`` identifies a row uniquely and no size term is needed.

Reading the output
------------------
Results are summarised at the test-fold level. Within a
``(test.subset, fold)``, all three train sources share a test set and
their differences move together, so the fold is the independent unit
and the 60 splits carry 20 clusters of information. A per-split
tolerance band would read fold-level sampling noise as disagreement and
is deliberately not reported.

Run from the repository root::

    uv run python analyses/run_glmnet_replication.py --full-only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from nsch_ml.soak import assign_folds, iter_soak_splits

# scikit-learn 1.8 is mid-migration on the penalty/l1_ratio API and emits
# FutureWarnings for spellings that still work. This is a hand-run analysis
# script, not library code; the noise buries the results.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.*")
# ConvergenceWarning subclasses UserWarning, so the filter above hides it. A
# solver that stops quietly at max_iter yields plausible coefficients and no
# signal, which is the failure mode least likely to be noticed.
warnings.filterwarnings("always", category=ConvergenceWarning)

DEFAULT_FIXTURE = Path.home() / "Documents/NAU/Grad/Research/ADSI/soak_fixture"
DEFAULT_REFERENCE = (
    Path.home()
    / "Documents/NAU/Grad/Research/ADSI/reproduce-soak-nsch"
    / "results/seed-variation/NSCH_seed1.csv"
)
N_FOLDS = 10
SEED = 1
EXPECTED_ROWS = 46010
# glmnet walks ~100 lambdas over two or three decades. sklearn's integer Cs
# spans 1e-4 to 1e4, so 12 points leaves cells ~5x wide -- far too coarse to
# see the penalty shift between training on 16k rows and 41k.
N_CS = np.logspace(-4, 0, 60)
INNER_CV = 5


def select_rules(cv: LogisticRegressionCV) -> tuple[float, float, int, int]:
    """Return (C_min, C_1se, index_min, index_1se) from the CV score grid.

    ``scores_`` is (n_folds, n_Cs), higher is better. The 1se rule takes
    the strongest penalty -- smallest C -- whose mean score is still
    within one standard error of the best mean.
    """
    raw = next(iter(cv.scores_.values()))
    scores = np.asarray(raw, dtype=np.float64)
    while scores.ndim > 2:
        scores = scores[..., 0]
    mean = scores.mean(axis=0)
    sem = scores.std(axis=0, ddof=1) / np.sqrt(scores.shape[0])
    best = int(np.argmax(mean))
    eligible = np.flatnonzero(mean >= mean[best] - sem[best])
    idx_1se = int(eligible.min())
    return float(cv.Cs_[best]), float(cv.Cs_[idx_1se]), best, idx_1se


def refit_kwargs(lasso: bool, l1_solver: str, intercept_scaling: float) -> dict[str, Any]:
    """Penalty and solver settings for a single fixed-C refit.

    scikit-learn 1.8 deprecated `penalty` in favour of `l1_ratio`, so the
    mixture is given as a ratio only: 1.0 is lasso, 0.0 is ridge. Passing both
    spellings produced an inconsistency warning and left it unclear which one
    the estimator honoured.

    liblinear and saga optimize the same L1 objective by different means,
    coordinate descent against a stochastic method. glmnet uses coordinate
    descent, so liblinear is the closer analogue and, at this problem size,
    about five times faster.

    `intercept_scaling` applies to liblinear alone, which penalizes the
    intercept where glmnet and saga do not. Larger values shrink that penalty
    toward zero; see analyses/probe_intercept_scaling.py for what it costs
    here.
    """
    if not lasso:
        return {"solver": "lbfgs", "l1_ratio": 0.0}
    if l1_solver == "liblinear":
        return {
            "solver": "liblinear",
            "l1_ratio": 1.0,
            "intercept_scaling": intercept_scaling,
        }
    return {"solver": "saga", "l1_ratio": 1.0}


def search_kwargs(lasso: bool, l1_solver: str, intercept_scaling: float) -> dict[str, Any]:
    """The same settings for LogisticRegressionCV, which spells the ratio as a list."""
    settings = refit_kwargs(lasso, l1_solver, intercept_scaling)
    ratio = settings.pop("l1_ratio")
    settings["l1_ratios"] = [ratio]
    return settings


def predict_at(
    c: float,
    x_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.int64],
    x_test: npt.NDArray[np.float64],
    lasso: bool,
    l1_solver: str,
    intercept_scaling: float,
) -> npt.NDArray[np.float64]:
    """Refit at a fixed C and return positive-class probabilities."""
    kwargs: dict[str, Any] = {"C": c, "max_iter": 5000}
    kwargs |= refit_kwargs(lasso, l1_solver, intercept_scaling)
    clf = LogisticRegression(**kwargs)
    clf.fit(x_train, y_train)
    return np.asarray(clf.predict_proba(x_test)[:, 1], dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-only", action="store_true")
    ap.add_argument("--lasso", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N splits")
    ap.add_argument("--out", default="analyses/glmnet_replication.csv")
    ap.add_argument(
        "--l1-solver",
        choices=("saga", "liblinear"),
        default="saga",
        help="solver for the lasso path; ignored for ridge, which uses lbfgs",
    )
    ap.add_argument(
        "--save-predictions",
        default=None,
        help="write one row per held-out child per split to this path",
    )
    ap.add_argument(
        "--intercept-scaling",
        type=float,
        default=1.0,
        help=(
            "liblinear only; larger values shrink the penalty liblinear places "
            "on the intercept, which glmnet does not penalize at all"
        ),
    )
    args = ap.parse_args()

    root = Path(os.environ.get("NSCH_SOAK_FIXTURE", DEFAULT_FIXTURE))
    ref_path = Path(os.environ.get("NSCH_SOAK_REFERENCE", DEFAULT_REFERENCE))
    print(f"fixture:   {root}")
    print(f"reference: {ref_path}")
    if not root.is_dir() or not ref_path.is_file():
        print("FAIL: fixture or reference not found", file=sys.stderr)
        return 2

    folds = pl.read_csv(root / "nsch_autism_folds.csv")
    design = pl.read_csv(root / "data_Classif" / "NSCH_autism.csv")
    reference = pl.read_csv(ref_path)
    if design.height != EXPECTED_ROWS:
        print("FAIL: unexpected row count", file=sys.stderr)
        return 2

    feature_cols = [c for c in design.columns if c not in ("survey_year", "y")]
    x_all = design.select(feature_cols).to_numpy().astype(np.float64)
    y_all = (design["y"].to_numpy().astype(str) == "Yes").astype(np.int64)
    subset = folds["test.subset"].cast(pl.Utf8).to_numpy()
    print(f"design {design.shape}   features {len(feature_cols)}")
    penalty_name = "lasso" if args.lasso else "ridge"
    settings = refit_kwargs(args.lasso, args.l1_solver, args.intercept_scaling)
    scaling_note = ""
    if "intercept_scaling" in settings:
        scaling_note = f"   intercept_scaling={args.intercept_scaling:g}"
    print(
        f"penalty: {penalty_name}   solver: {settings['solver']}   "
        f"inner_cv={INNER_CV}{scaling_note}"
    )
    print(f"Cs: {len(N_CS)} points from {N_CS.min():.1e} to {N_CS.max():.1e}")

    ref: dict[tuple[str, str, int], tuple[float, float]] = {}
    for s, src, f, auc, acc in reference.select(
        ["test.subset", "train.subsets", "test.fold", "classif.auc", "classif.acc"]
    ).iter_rows():
        ref[(str(s), str(src), int(f))] = (float(auc), float(acc))
    print(f"reference: {reference.height} rows -> {len(ref)} unique (subset, source, fold)")
    if len(ref) != reference.height:
        print("  note: reference had duplicate keys; last row won")

    fold_ids = assign_folds(
        subset=subset,
        outcome=design["y"],
        n_folds=N_FOLDS,
        precomputed=folds["fold"],
    )
    splits = list(
        iter_soak_splits(
            fold_ids=fold_ids,
            subset=subset,
            outcome=design["y"],
            sizes=0,
            seed=SEED,
        )
    )
    if args.full_only:
        splits = [s for s in splits if not s.downsampled]
    if args.limit:
        splits = splits[: args.limit]
    print(f"splits to fit: {len(splits)}\n")

    rows = []
    prediction_rows: list[dict[str, object]] = []
    t_start = time.perf_counter()
    for i, split in enumerate(splits, 1):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_all[split.train_idx])
        x_test = scaler.transform(x_all[split.test_idx])
        y_train, y_test = y_all[split.train_idx], y_all[split.test_idx]

        cv_kwargs: dict[str, Any] = {
            "Cs": N_CS,
            "cv": INNER_CV,
            "scoring": "neg_log_loss",
            "max_iter": 5000,
            "n_jobs": -1,
        }
        cv_kwargs |= search_kwargs(args.lasso, args.l1_solver, args.intercept_scaling)
        cv = LogisticRegressionCV(**cv_kwargs)
        cv.fit(x_train, y_train)
        c_min, c_1se, i_min, i_1se = select_rules(cv)

        rec: dict[str, object] = {
            "test_subset": split.test_subset,
            "train_source": split.train_source.value,
            "fold": split.fold,
            "downsampled": split.downsampled,
            "n_train": len(split.train_idx),
            "n_test": len(split.test_idx),
            "C_min": c_min,
            "C_1se": c_1se,
            "idx_min": i_min,
            "idx_1se": i_1se,
        }
        probability_of: dict[str, npt.NDArray[np.float64]] = {}
        for label, c in (("min", c_min), ("1se", c_1se)):
            prob = predict_at(
                c, x_train, y_train, x_test, args.lasso, args.l1_solver, args.intercept_scaling
            )
            probability_of[label] = prob
            rec[f"auc_{label}"] = roc_auc_score(y_test, prob)
            rec[f"acc_{label}"] = accuracy_score(y_test, (prob >= 0.5).astype(np.int64))

        if args.save_predictions:
            # row_id is 1-based to match the folds file and R's export, so the
            # two sides join without an offset correction.
            for position, design_row in enumerate(split.test_idx):
                prediction_rows.append(
                    {
                        "test_subset": split.test_subset,
                        "train_source": split.train_source.value,
                        "fold": split.fold,
                        "row_id": int(design_row) + 1,
                        "truth": int(y_test[position]),
                        "py_prob_min": float(probability_of["min"][position]),
                        "py_prob_1se": float(probability_of["1se"][position]),
                    }
                )

        r = ref.get((split.test_subset, split.train_source.value, split.fold))
        rec["r_auc"] = r[0] if r else None
        rec["r_acc"] = r[1] if r else None
        for label in ("min", "1se"):
            rec[f"d_auc_{label}"] = float(rec[f"auc_{label}"]) - r[0] if r else None  # type: ignore[arg-type]
        rows.append(rec)

        delta_txt = (
            f"  d_min {rec['d_auc_min']:+.5f}  d_1se {rec['d_auc_1se']:+.5f}"
            if r
            else "  [no R match]"
        )
        print(
            f"[{i:3d}/{len(splits)}] {split.test_subset} "
            f"{split.train_source.value:5s} f{split.fold:<2d} "
            f"{'down' if split.downsampled else 'full'}  "
            f"C {c_min:.4g}/{c_1se:.4g} (idx {i_min}/{i_1se})  "
            f"auc {rec['auc_min']:.5f}/{rec['auc_1se']:.5f}{delta_txt}"
        )

    out = pl.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_path)
    print(f"\nwrote {out_path}   ({time.perf_counter() - t_start:.0f}s total)")

    if args.save_predictions:
        predictions_path = Path(args.save_predictions)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(prediction_rows).write_csv(predictions_path)
        n_splits_written = len(
            {(row["test_subset"], row["train_source"], row["fold"]) for row in prediction_rows}
        )
        print(
            f"wrote {predictions_path}   "
            f"({len(prediction_rows)} rows over {n_splits_written} splits)"
        )

    matched = out.filter(pl.col("r_auc").is_not_null())
    print(f"\n== agreement ({matched.height} of {out.height} splits matched to R) ==")
    if matched.height == 0:
        print("no R matches -- check the reference join keys")
        return 1

    # Differences cluster by test fold: within a (subset, fold), all three
    # train sources share a test set and move together. The fold is therefore
    # the independent unit, not the split, and 60 rows carry 20 clusters of
    # information. A per-split tolerance band would read fold-level sampling
    # noise as disagreement, so it is deliberately not reported.
    for label in ("min", "1se"):
        per_fold = (
            matched.group_by(["test_subset", "fold"])
            .agg(pl.col(f"d_auc_{label}").mean().alias("d"))
            .sort(["test_subset", "fold"])
        )
        d = per_fold["d"].to_numpy()
        n = len(d)
        mean = float(d.mean())
        sd = float(d.std(ddof=1)) if n > 1 else float("nan")
        sem = sd / np.sqrt(n) if n > 1 else float("nan")
        t_stat = mean / sem if sem else float("nan")
        print(
            f"  lambda.{label:3s}  folds {n}  mean {mean:+.5f}  "
            f"median {np.median(d):+.5f}  sd {sd:.5f}  sem {sem:.5f}  "
            f"t {t_stat:+.2f}  max|d| {np.abs(d).max():.5f}"
        )

    print("\n  Unit of analysis is the test fold, not the split. A mean near")
    print("  zero with |t| well under 2 means no detectable systematic")
    print("  difference in discrimination between the R and Python fits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
