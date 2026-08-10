"""Does the Python port reproduce the R analysis?

Judged against the margins in ``docs/equivalence-margin.md``, which were
committed before this script existed. The margins are repeated here as
constants so a reader can see what is being tested without opening another
file, but the document is the authority; if the two ever disagree, the
document wins and this script is wrong.

    Spearman of held-out scores, per split   at least 0.95      gated
    Fold-level AUC, mean absolute difference at most 0.002      gated
    Probability-scale MAD                    at most 0.01       gated, provisional
    Coefficient mean absolute difference     reported, not gated
    Feature selection agreement              reported, not gated

Coefficients are reported rather than gated because three R runs differing
only in the inner cross-validation seed disagree with each other by 0.004
mean absolute and by as much as 0.377 on a single weight. Gating on that
would measure the lasso's instability rather than the port.

The probability margin is marked provisional in the document because it was
set without a reference measurement. R's probabilities exist for one seed
only, so there is no R-internal spread to anchor it against.

Three label conventions matter here and none is visible in an AUC: the
design's outcome is "Yes"/"No" rather than 1/0, mlr3 assigned "No" as
positive so R's raw probabilities are one minus the outcome probability, and
scikit-learn treats the lexicographically last string label as positive. The
R predictions this script reads have already been corrected by
``analyses/repair_r_predictions.py``; the check that they were is that their
mean lands on the outcome prevalence.

Run from the repository root::

    uv run python analyses/prediction_equivalence.py \\
        --r-predictions      $REF/NSCH_seed1_predictions_repaired.csv \\
        --python-predictions $REF/python_lasso_seed1_is100_predictions.csv \\
        --python-auc         analyses/glmnet_replication_lasso_seed1_is100.csv

where $REF is the predictions directory of the reproduce-soak-nsch checkout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from scipy import stats

# From docs/equivalence-margin.md. That document is the authority.
SPEARMAN_FLOOR = 0.95
AUC_MEAN_ABS_CEILING = 0.002
PROBABILITY_MAD_CEILING = 0.01

SPLIT_KEY = ["test_subset", "train_source", "fold"]
EXPECTED_SPLITS = 60
EXPECTED_ROWS = 138030
# R's cv_glmnet predicts at lambda.1se, so that is the like-for-like column.
# lambda.min is reported alongside because it costs nothing and shows whether
# the selection rule matters more or less than the implementation gap.
GATED_RULE = "1se"


def load_predictions(path: Path, probability_columns: dict[str, str]) -> pl.DataFrame:
    """Read a predictions file and normalise the join key and column names."""
    frame = pl.read_csv(path).with_columns(pl.col("test_subset").cast(pl.Utf8))
    missing = [name for name in probability_columns if name not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing {missing}")
    return frame.select(
        *SPLIT_KEY,
        "row_id",
        "truth",
        *[pl.col(source).alias(target) for source, target in probability_columns.items()],
    )


def summarise(values: list[float]) -> str:
    array = np.asarray(values, dtype=np.float64)
    return f"mean {array.mean():.6f}  median {np.median(array):.6f}  worst {array.max():.6f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r-predictions", required=True)
    parser.add_argument("--python-predictions", required=True)
    parser.add_argument("--python-auc", required=True, help="the per-split CSV with r_auc")
    args = parser.parse_args()

    paths = {
        "R predictions": Path(args.r_predictions),
        "Python predictions": Path(args.python_predictions),
        "Python AUC": Path(args.python_auc),
    }
    for label, path in paths.items():
        if not path.is_file():
            print(f"FAIL: {label} not found at {path}", file=sys.stderr)
            return 2

    r_frame = load_predictions(paths["R predictions"], {"r_prob": "r_prob"})
    python_frame = load_predictions(
        paths["Python predictions"],
        {"py_prob_1se": "py_prob_1se", "py_prob_min": "py_prob_min"},
    )
    print(f"R predictions     {r_frame.height} rows")
    print(f"Python predictions {python_frame.height} rows")

    paired = r_frame.join(python_frame, on=[*SPLIT_KEY, "row_id"], how="inner", suffix="_py")
    print(f"joined on split and row_id: {paired.height} rows")
    if paired.height != r_frame.height or paired.height != python_frame.height:
        print(
            "FAIL: the two files do not cover the same rows. A partial comparison "
            "would look like agreement on whatever happened to overlap.",
            file=sys.stderr,
        )
        return 1
    if paired.height != EXPECTED_ROWS:
        print(f"FAIL: expected {EXPECTED_ROWS} rows, got {paired.height}", file=sys.stderr)
        return 1

    # The truth columns come from different files by different routes, so they
    # are an independent check that the join lined up the same children.
    disagreeing = int((paired["truth"] != paired["truth_py"]).sum())
    if disagreeing:
        print(f"FAIL: {disagreeing} rows disagree about the outcome", file=sys.stderr)
        return 1
    prevalence = cast("float", paired["truth"].mean())
    print(f"outcome prevalence {prevalence:.4f}, truth agrees on every row")

    splits = paired.select(SPLIT_KEY).unique().sort(SPLIT_KEY)
    if splits.height != EXPECTED_SPLITS:
        print(f"FAIL: {splits.height} splits, expected {EXPECTED_SPLITS}", file=sys.stderr)
        return 1

    # ------------------------------------------------ per-split quantities
    print("\n" + "=" * 92)
    print("Per split: rank agreement and probability difference")
    print("=" * 92)
    print(
        f"{'subset':>8}{'source':>8}{'fold':>6}{'n':>7}"
        f"{'rho 1se':>10}{'MAD 1se':>10}{'bias 1se':>11}"
        f"{'rho min':>10}{'MAD min':>10}   gated"
    )

    spearman_1se: list[float] = []
    mad_1se: list[float] = []
    bias_1se: list[float] = []
    spearman_min: list[float] = []
    mad_min: list[float] = []
    failing_splits: list[str] = []

    for key in splits.iter_rows(named=True):
        cell = paired.filter(
            (pl.col("test_subset") == key["test_subset"])
            & (pl.col("train_source") == key["train_source"])
            & (pl.col("fold") == key["fold"])
        )
        r_probability = cell["r_prob"].to_numpy()
        rho_values = {}
        mad_values = {}
        for rule in ("1se", "min"):
            python_probability = cell[f"py_prob_{rule}"].to_numpy()
            rho_values[rule] = float(stats.spearmanr(r_probability, python_probability).statistic)
            mad_values[rule] = float(np.abs(python_probability - r_probability).mean())
        signed_bias = float((cell["py_prob_1se"].to_numpy() - r_probability).mean())

        spearman_1se.append(rho_values["1se"])
        mad_1se.append(mad_values["1se"])
        bias_1se.append(signed_bias)
        spearman_min.append(rho_values["min"])
        mad_min.append(mad_values["min"])

        passes = (
            rho_values[GATED_RULE] >= SPEARMAN_FLOOR
            and mad_values[GATED_RULE] <= PROBABILITY_MAD_CEILING
        )
        label = f"{key['test_subset']} {key['train_source']} f{key['fold']}"
        if not passes:
            failing_splits.append(label)
        print(
            f"{key['test_subset']:>8}{key['train_source']:>8}{key['fold']:>6}{cell.height:>7}"
            f"{rho_values['1se']:>10.5f}{mad_values['1se']:>10.6f}{signed_bias:>+11.6f}"
            f"{rho_values['min']:>10.5f}{mad_values['min']:>10.6f}   "
            f"{'yes' if passes else 'NO'}"
        )

    # ------------------------------------------------------ fold-level AUC
    auc_frame = pl.read_csv(paths["Python AUC"]).filter(
        ~pl.col("downsampled") & pl.col("r_auc").is_not_null()
    )
    auc_differences = np.abs(
        auc_frame[f"auc_{GATED_RULE}"].to_numpy() - auc_frame["r_auc"].to_numpy()
    )
    auc_mean_abs = float(auc_differences.mean())

    # ------------------------------------------------------------- verdict
    print("\n" + "=" * 92)
    print("Against the margins in docs/equivalence-margin.md")
    print("=" * 92)
    worst_spearman = float(np.min(spearman_1se))
    worst_mad = float(np.max(mad_1se))
    checks = [
        (
            f"Spearman per split, {GATED_RULE}",
            f"worst {worst_spearman:.5f}",
            f"at least {SPEARMAN_FLOOR}",
            worst_spearman >= SPEARMAN_FLOOR,
        ),
        (
            f"Probability MAD per split, {GATED_RULE}",
            f"worst {worst_mad:.6f}",
            f"at most {PROBABILITY_MAD_CEILING}",
            worst_mad <= PROBABILITY_MAD_CEILING,
        ),
        (
            f"Fold-level AUC, mean absolute, {GATED_RULE}",
            f"{auc_mean_abs:.6f}",
            f"at most {AUC_MEAN_ABS_CEILING}",
            auc_mean_abs <= AUC_MEAN_ABS_CEILING,
        ),
    ]
    print(f"{'quantity':<40}{'observed':>22}{'margin':>18}{'':>6}")
    for name, observed, margin, passed in checks:
        print(f"{name:<40}{observed:>22}{margin:>18}{'  PASS' if passed else '  FAIL':>6}")

    print("\nreported, not gated")
    print(
        f"  Spearman, 1se        {summarise([1 - value for value in spearman_1se])}   (as 1 - rho)"
    )
    print(f"  Probability MAD, min {summarise(mad_min)}")
    print(
        f"  Signed bias, 1se     mean {np.mean(bias_1se):+.6f}, "
        f"so Python reads {'high' if np.mean(bias_1se) > 0 else 'low'} on average"
    )

    if failing_splits:
        print(f"\nsplits failing a gated check: {len(failing_splits)}")
        for label in failing_splits:
            print(f"  {label}")

    all_passed = all(passed for _, _, _, passed in checks) and not failing_splits
    print("\n" + "=" * 92)
    if all_passed:
        print("PASS on every gated quantity, on every split.")
        print("The port reproduces the R analysis to the standard fixed in advance.")
    else:
        print("FAIL on at least one gated quantity.")
        print("The margins were committed before this ran and are not to be revised now.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
