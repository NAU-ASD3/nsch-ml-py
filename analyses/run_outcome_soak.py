"""Run the SOAK design for one extension outcome on one matrix.

This is `run_glmnet_replication.py` pointed at a question whose answer is not
already known. The learner is imported from that script rather than restated,
so both use one definition of the model and the validated replication stays
untouched.

What is different from the replication, and why:

*It takes an outcome and a matrix.* Both come from `outcomes.py`, which owns
the positive-class rule and the columns that must be removed. The fold draw
reads the same definitions, so a run cannot be stratified on one rule and
fitted on another.

*It keeps the downsampled splits.* The replication filtered them out. Here they
are the point: `Other` trains on up to 5.4 times as many children as `Same`, so
the full-size Other-minus-Same contrast measures training-set size as much as
anything about the periods. See docs/extension-analysis-plan.md.

*It records calibration.* AUC is invariant to any monotone transformation of
the predicted probabilities, so a model can rank children perfectly while badly
misstating how many of them lack care. With prevalence moving from 9% to 14%
across periods, that is the property most likely to separate the training
strategies, and AUC alone cannot see it.

*It saves what a later question would otherwise need a refit to answer.*
Per-child predictions and per-split non-zero coefficients are written out, so
stability selection, recalibration, and any metric we have not thought of yet
can be computed without spending the compute again.

*It does not require an R reference.* There is none for these outcomes. Pass
`--reference` to compare against one anyway, which is how a run against the
replication's own outcome is checked.

Check the design before spending the compute::

    uv run python analyses/run_outcome_soak.py --matrix service_use \\
        --data "$MONSOON_OLD/2016_2023_ServiceUse.csv" \\
        --outcome ed_any \\
        --folds analyses/folds/service_ed_any_folds.csv \\
        --dry-run

Then drop `--dry-run` to fit.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from outcomes import matrix_or_exit
from run_glmnet_replication import INNER_CV, N_CS, refit_kwargs, search_kwargs, select_rules
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from nsch_ml.soak import assign_folds, iter_soak_splits

TRACKED_PACKAGES = ("nsch-ml", "polars", "numpy", "scikit-learn", "scipy")
DEFAULT_SEED = 1
SELECTION_RULES = ("min", "1se")
# Probabilities are clipped before the logit for the calibration fit. Anything
# closer to the boundary than this contributes an unbounded logit and would
# dominate the slope on its own.
PROBABILITY_CLIP = 1e-6
RULE = "=" * 78


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def git_commit() -> str:
    """The commit this ran at, so a result can be tied to the code that made it."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def calibration_fit(
    truth: npt.NDArray[np.int64], probability: npt.NDArray[np.float64]
) -> tuple[float | None, float | None]:
    """Slope and intercept of the outcome regressed on the predicted logit.

    A perfectly calibrated model gives slope 1 and intercept 0. A slope below
    one means the predictions are too extreme, above one that they are too
    timid, and the intercept carries any overall shift. Fitted unpenalized: C
    is set enormous rather than passing ``penalty=None``, whose spelling is
    mid-deprecation in this scikit-learn.
    """
    if len(np.unique(truth)) < 2:
        return None, None
    clipped = np.clip(probability, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=2000)
    model.fit(logit, truth)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def load_folds(
    folds_path: Path, matrix: pl.DataFrame, subset_column: str
) -> tuple[pl.DataFrame | None, list[str]]:
    """Read the fold assignment and check it describes this matrix, row for row."""
    folds = pl.read_csv(folds_path)
    missing = [c for c in ("row_id", "test.subset", "fold") if c not in folds.columns]
    if missing:
        return None, [f"the fold file has no {name!r} column" for name in missing]
    if folds.height != matrix.height:
        return None, [
            f"the fold file has {folds.height} rows and the matrix {matrix.height}; "
            "they cannot describe the same children"
        ]
    misaligned = int(
        (folds["test.subset"].cast(pl.Utf8) != matrix[subset_column].cast(pl.Utf8)).sum()
    )
    if misaligned:
        return folds, [
            f"{misaligned} rows carry a different subset label in the fold file than in "
            "the matrix, so the assignment does not line up with this matrix"
        ]
    return folds, []


def check_fold_provenance(folds_path: Path, matrix_md5: str, expected_outcome: str) -> list[str]:
    """Confirm the folds were drawn against this matrix, for this outcome.

    A fold file is three columns of integers and gives no sign of being the
    wrong ones. The provenance record beside it does, so if it exists it is
    checked rather than trusted.
    """
    provenance_path = folds_path.with_suffix(".provenance.csv")
    if not provenance_path.is_file():
        return [
            f"no provenance record at {provenance_path}. Folds without provenance cannot "
            "be tied to a matrix or a seed; redraw them with draw_outcome_folds.py."
        ]
    record = pl.read_csv(provenance_path)
    fields = dict(zip(record["field"].to_list(), record["value"].to_list(), strict=True))
    problems: list[str] = []
    if fields.get("matrix_md5") != matrix_md5:
        problems.append(
            f"the folds were drawn against a matrix with md5 {fields.get('matrix_md5')}, "
            f"but this matrix is {matrix_md5}"
        )
    if fields.get("outcome") != expected_outcome:
        problems.append(
            f"the folds were drawn for outcome {fields.get('outcome')!r}, not {expected_outcome!r}"
        )
    return problems


def describe_inventory(splits: list[Any]) -> pl.DataFrame:
    """One row per (subset, source, size variant): how many splits, how big."""
    return (
        pl.DataFrame(
            [
                {
                    "test_subset": split.test_subset,
                    "train_source": split.train_source.value,
                    "downsampled": split.downsampled,
                    "n_train": len(split.train_idx),
                    "n_test": len(split.test_idx),
                }
                for split in splits
            ]
        )
        .group_by(["test_subset", "train_source", "downsampled"])
        .agg(
            pl.len().alias("splits"),
            pl.col("n_train").min().alias("train_min"),
            pl.col("n_train").max().alias("train_max"),
            pl.col("n_test").sum().alias("test_total"),
        )
        .sort(["test_subset", "train_source", "downsampled"])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, help="fixture or service_use")
    parser.add_argument("--data", required=True, help="path to the matrix CSV")
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--folds", required=True, help="fold assignment CSV")
    parser.add_argument("--out", default=None, help="per-split metrics CSV")
    parser.add_argument("--run-dir", default="analyses/runs", help="predictions and coefficients")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sizes", type=int, choices=(-1, 0), default=0)
    parser.add_argument("--ridge", action="store_true", help="ridge instead of the planned lasso")
    parser.add_argument("--l1-solver", choices=("saga", "liblinear"), default="liblinear")
    parser.add_argument("--intercept-scaling", type=float, default=100.0)
    parser.add_argument("--n-jobs", type=int, default=-1, help="lower when running in parallel")
    parser.add_argument("--limit", type=int, default=0, help="stop after N splits")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--reference", default=None, help="optional R results to compare against")
    parser.add_argument(
        "--allow-missing-provenance",
        action="store_true",
        help=(
            "proceed when the fold file has no provenance record beside it. Only for "
            "externally drawn folds, such as the R assignment the replication used, "
            "whose provenance lives with the fixture that produced it"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="describe the design and stop")
    args = parser.parse_args()

    data_path, folds_path = Path(args.data), Path(args.folds)
    for label, path in (("matrix", data_path), ("folds", folds_path)):
        if not path.is_file():
            print(f"REFUSED: no {label} at {path}")
            return 1

    spec = matrix_or_exit(args.matrix)
    outcome = spec.outcome_or_exit(args.outcome)
    lasso = not args.ridge

    matrix = pl.read_csv(data_path)
    matrix_md5 = file_md5(data_path)

    print(RULE)
    print(f"Matrix    {data_path}")
    print(f"          {spec.label}")
    print(f"md5       {matrix_md5}")
    print(f"Outcome   {outcome.key}: {outcome.label}")
    print(f"Folds     {folds_path}")

    # Every column named for removal must exist. A drop list that silently
    # matches nothing would leave the outcome sitting among the features, and
    # the run would finish with an implausibly good result that looks like
    # success rather than like a mistake.
    problems = [
        f"{name!r} is named for removal but is not in the matrix"
        for name in outcome.drop_columns
        if name not in matrix.columns
    ]
    if spec.subset_column not in matrix.columns:
        problems.append(f"the matrix has no {spec.subset_column!r} column to split on")
    if not problems:
        provenance_problems = check_fold_provenance(folds_path, matrix_md5, outcome.folds_key)
        if provenance_problems and args.allow_missing_provenance:
            print("\nWARNING: proceeding without a fold provenance check.")
            for problem in provenance_problems:
                print(f"  {problem}")
            print("  Provenance for these folds must be recorded elsewhere.\n")
        else:
            problems += provenance_problems
    folds: pl.DataFrame | None = None
    if not problems:
        folds, fold_problems = load_folds(folds_path, matrix, spec.subset_column)
        problems += fold_problems
    if problems or folds is None:
        print(RULE)
        for problem in problems:
            print(f"REFUSED: {problem}")
        return 1

    y_all = matrix.select(outcome.positive.alias("p"))["p"].cast(pl.Int64).to_numpy()
    y_all = y_all.astype(np.int64)
    if len(np.unique(y_all)) < 2:
        print(f"REFUSED: the outcome is constant across all {matrix.height} rows")
        return 1

    feature_names = spec.feature_columns(matrix.columns, outcome)
    features = matrix.select(feature_names)
    non_numeric = [
        name
        for name, dtype in zip(features.columns, features.dtypes, strict=True)
        if not dtype.is_numeric()
    ]
    if non_numeric:
        print(RULE)
        print(
            f"REFUSED: {len(non_numeric)} feature columns are not numeric, so they cannot "
            f"be modelled as they stand: {', '.join(non_numeric[:8])}"
        )
        return 1
    null_total = int(features.null_count().sum_horizontal().item())
    if null_total:
        print(RULE)
        print(
            f"REFUSED: the feature matrix holds {null_total} missing values. The plan "
            "requires the missing-data encoding to be settled before these runs are "
            "interpreted, and imputing silently here would pre-empt that decision."
        )
        return 1

    x_all = features.to_numpy().astype(np.float64)
    subset = matrix[spec.subset_column].cast(pl.Utf8).to_numpy()

    print(f"Rows      {matrix.height}, positives {int(y_all.sum())} ({y_all.mean():.1%})")
    print(f"Features  {len(feature_names)} of {matrix.width} columns")
    print(f"Removed   {', '.join(outcome.drop_columns) or 'nothing'}")
    print(
        f"Learner   {'lasso' if lasso else 'ridge'}, "
        f"{args.l1_solver if lasso else 'lbfgs'}, inner_cv={INNER_CV}, "
        f"{len(N_CS)} penalties"
    )

    # Series.max() is typed as a union spanning every dtype polars can hold, so
    # it does not narrow to int. Going through to_list() keeps this honest
    # without a cast that would suppress a real error.
    n_folds = max(int(value) for value in folds["fold"].to_list())
    fold_ids = assign_folds(
        subset=subset,
        outcome=y_all,
        n_folds=n_folds,
        precomputed=folds["fold"],
    )
    splits = list(
        iter_soak_splits(
            fold_ids=fold_ids,
            subset=subset,
            outcome=y_all,
            sizes=args.sizes,
            seed=args.seed,
        )
    )

    # A source already at the target size is not duplicated by the splitter, so
    # `downsampled` alone does not identify the equal-size arm: where Same is
    # the smallest source, Same's equal-size arm is its full split. Group sizes
    # settle which is which without re-deriving the size formula.
    group_sizes = Counter(
        (split.test_subset, split.fold, split.train_source.value) for split in splits
    )

    def equal_size_flag(split: Any) -> bool | None:
        if args.sizes != 0:
            return None
        key = (split.test_subset, split.fold, split.train_source.value)
        return bool(split.downsampled or group_sizes[key] == 1)

    print(RULE)
    print(f"Design    {len(splits)} splits, sizes={args.sizes}")
    for row in describe_inventory(splits).iter_rows(named=True):
        variant = "equal-size" if row["downsampled"] else "full"
        train_range = (
            f"{row['train_min']:,}"
            if row["train_min"] == row["train_max"]
            else f"{row['train_min']:,} to {row['train_max']:,}"
        )
        print(
            f"  {row['test_subset']:>8}  {row['train_source']:<5} {variant:<10} "
            f"{row['splits']:>3} splits   train {train_range}"
        )

    if args.dry_run:
        print(RULE)
        print("Dry run: nothing was fitted. Check the split count and the training")
        print("sizes above against the plan before running this for real.")
        return 0

    if args.limit:
        splits = splits[: args.limit]
        print(f"  limited to the first {len(splits)} splits")

    out_path = (
        Path(args.out) if args.out else Path("analyses/results") / f"{spec.key}_{outcome.key}.csv"
    )
    run_dir = Path(args.run_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / f"{spec.key}_{outcome.key}_predictions.csv"
    coefficients_path = run_dir / f"{spec.key}_{outcome.key}_coefficients.csv"
    provenance_path = out_path.with_suffix(".provenance.csv")

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[tuple[Any, ...]] = []
    coefficient_rows: list[tuple[Any, ...]] = []

    prediction_schema = [
        ("test_subset", pl.Utf8),
        ("train_source", pl.Utf8),
        ("fold", pl.Int64),
        ("downsampled", pl.Boolean),
        ("row_id", pl.Int64),
        ("truth", pl.Int64),
        ("prob_min", pl.Float64),
        ("prob_1se", pl.Float64),
    ]
    coefficient_schema = [
        ("test_subset", pl.Utf8),
        ("train_source", pl.Utf8),
        ("fold", pl.Int64),
        ("downsampled", pl.Boolean),
        ("rule", pl.Utf8),
        ("feature", pl.Utf8),
        ("coefficient", pl.Float64),
    ]

    def write_outputs() -> None:
        """Write all three artifacts. Called periodically so a crash costs minutes."""
        pl.DataFrame(metric_rows).write_csv(out_path)
        if prediction_rows:
            pl.DataFrame(prediction_rows, schema=prediction_schema, orient="row").write_csv(
                predictions_path
            )
        if coefficient_rows:
            pl.DataFrame(coefficient_rows, schema=coefficient_schema, orient="row").write_csv(
                coefficients_path
            )

    print(RULE)
    started = datetime.now(UTC)
    clock = time.perf_counter()
    for position, split in enumerate(splits, 1):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_all[split.train_idx])
        x_test = scaler.transform(x_all[split.test_idx])
        y_train, y_test = y_all[split.train_idx], y_all[split.test_idx]
        single_class_test = len(np.unique(y_test)) < 2

        search: dict[str, Any] = {
            "Cs": N_CS,
            "cv": INNER_CV,
            "scoring": "neg_log_loss",
            "max_iter": 5000,
            "n_jobs": args.n_jobs,
        }
        search |= search_kwargs(lasso, args.l1_solver, args.intercept_scaling)
        cv = LogisticRegressionCV(**search)
        cv.fit(x_train, y_train)
        c_min, c_1se, i_min, i_1se = select_rules(cv)

        record: dict[str, Any] = {
            "test_subset": split.test_subset,
            "train_source": split.train_source.value,
            "fold": split.fold,
            "downsampled": split.downsampled,
            "is_equal_size": equal_size_flag(split),
            "n_train": len(split.train_idx),
            "n_test": len(split.test_idx),
            "n_train_positive": int(y_train.sum()),
            "n_test_positive": int(y_test.sum()),
            "test_prevalence": float(y_test.mean()),
            "C_min": c_min,
            "C_1se": c_1se,
            "idx_min": i_min,
            "idx_1se": i_1se,
        }

        probabilities: dict[str, npt.NDArray[np.float64]] = {}
        for label, c in zip(SELECTION_RULES, (c_min, c_1se), strict=True):
            fit_args: dict[str, Any] = {"C": c, "max_iter": 5000}
            fit_args |= refit_kwargs(lasso, args.l1_solver, args.intercept_scaling)
            model = LogisticRegression(**fit_args)
            model.fit(x_train, y_train)
            probability = np.asarray(model.predict_proba(x_test)[:, 1], dtype=np.float64)
            probabilities[label] = probability

            accuracy = float(accuracy_score(y_test, (probability >= 0.5).astype(np.int64)))
            slope, intercept = calibration_fit(y_test, probability)
            record[f"auc_{label}"] = (
                None if single_class_test else float(roc_auc_score(y_test, probability))
            )
            record[f"acc_{label}"] = accuracy
            record[f"percent_error_{label}"] = 100.0 * (1.0 - accuracy)
            record[f"brier_{label}"] = float(brier_score_loss(y_test, probability))
            record[f"calibration_slope_{label}"] = slope
            record[f"calibration_intercept_{label}"] = intercept

            coefficients = model.coef_[0]
            non_zero = np.flatnonzero(coefficients)
            record[f"n_nonzero_{label}"] = len(non_zero)
            coefficient_rows.extend(
                (
                    split.test_subset,
                    split.train_source.value,
                    split.fold,
                    split.downsampled,
                    label,
                    feature_names[int(index)],
                    float(coefficients[int(index)]),
                )
                for index in non_zero
            )

        prediction_rows.extend(
            (
                split.test_subset,
                split.train_source.value,
                split.fold,
                split.downsampled,
                int(design_row) + 1,
                int(y_test[offset]),
                float(probabilities["min"][offset]),
                float(probabilities["1se"][offset]),
            )
            for offset, design_row in enumerate(split.test_idx)
        )

        metric_rows.append(record)
        elapsed = time.perf_counter() - clock
        remaining = elapsed / position * (len(splits) - position)
        auc_text = f"{record['auc_1se']:.5f}" if record["auc_1se"] is not None else "  n/a  "
        slope_text = (
            f"{record['calibration_slope_1se']:.3f}"
            if record["calibration_slope_1se"] is not None
            else "  n/a"
        )
        print(
            f"[{position:3d}/{len(splits)}] {split.test_subset:>8} "
            f"{split.train_source.value:<5} f{split.fold:<2d} "
            f"{'equal' if split.downsampled else 'full ':<5} "
            f"n {len(split.train_idx):>6,}   auc {auc_text}   "
            f"brier {record['brier_1se']:.4f}   slope {slope_text}   "
            f"kept {record['n_nonzero_1se']:>3}   eta {remaining / 60:.0f}m"
        )
        if args.checkpoint_every and position % args.checkpoint_every == 0:
            write_outputs()

    write_outputs()
    total_seconds = time.perf_counter() - clock

    pl.DataFrame(
        {
            "field": [
                "started_utc",
                "finished_utc",
                "elapsed_seconds",
                "git_commit",
                "matrix_key",
                "matrix",
                "matrix_md5",
                "outcome",
                "outcome_label",
                "folds",
                "folds_md5",
                "folds_drawn_for",
                "dropped_from_features",
                "n_features",
                "rows",
                "positives",
                "sizes",
                "seed",
                "n_splits",
                "penalty",
                "solver",
                "intercept_scaling",
                "inner_cv",
                "n_penalties",
                "python",
                *TRACKED_PACKAGES,
            ],
            "value": [
                started.isoformat(timespec="seconds"),
                datetime.now(UTC).isoformat(timespec="seconds"),
                f"{total_seconds:.0f}",
                git_commit(),
                spec.key,
                str(data_path),
                matrix_md5,
                outcome.key,
                outcome.label,
                str(folds_path),
                file_md5(folds_path),
                outcome.folds_key,
                ", ".join(outcome.drop_columns) or "none",
                str(len(feature_names)),
                str(matrix.height),
                str(int(y_all.sum())),
                str(args.sizes),
                str(args.seed),
                str(len(splits)),
                "lasso" if lasso else "ridge",
                args.l1_solver if lasso else "lbfgs",
                f"{args.intercept_scaling:g}",
                str(INNER_CV),
                str(len(N_CS)),
                platform.python_version(),
                *[package_version(name) for name in TRACKED_PACKAGES],
            ],
        }
    ).write_csv(provenance_path)

    print(RULE)
    print(f"wrote {out_path}   ({len(metric_rows)} splits, {total_seconds / 60:.1f} minutes)")
    print(f"wrote {predictions_path}   ({len(prediction_rows):,} rows)")
    print(f"wrote {coefficients_path}   ({len(coefficient_rows):,} rows)")
    print(f"wrote {provenance_path}")

    if not args.reference:
        return 0

    reference_path = Path(args.reference)
    if not reference_path.is_file():
        print(f"\nreference not found at {reference_path}; skipping the comparison")
        return 0
    reference = (
        pl.read_csv(reference_path, infer_schema_length=20000)
        .filter(pl.col("learner_id") == "classif.cv_glmnet")
        .sort("n.train.groups", descending=True)
        .unique(subset=["test.subset", "train.subsets", "test.fold"], keep="first")
        .select(
            pl.col("test.subset").cast(pl.Utf8).alias("test_subset"),
            pl.col("train.subsets").alias("train_source"),
            pl.col("test.fold").alias("fold"),
            pl.col("classif.auc").alias("reference_auc"),
        )
    )
    joined = (
        pl.DataFrame(metric_rows)
        .filter(~pl.col("downsampled"))
        .join(reference, on=["test_subset", "train_source", "fold"], how="inner")
    )
    print(f"\n== compared to {reference_path.name} ==")
    if joined.height == 0:
        print("no rows matched; check the join keys")
        return 1
    difference = (joined["auc_1se"] - joined["reference_auc"]).to_numpy()
    print(
        f"  {joined.height} splits matched   mean {difference.mean():+.5f}   "
        f"max |d| {np.abs(difference).max():.5f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
