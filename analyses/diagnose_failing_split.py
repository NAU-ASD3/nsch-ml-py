"""Why does one fitted model miss the rank-agreement margin?

Two of 60 splits fall below the Spearman floor of 0.95: 2019 "same" fold 7 at
0.94057 and 2020 "other" fold 7 at 0.94106. Those are not two failures. For
any fold, "same" on 2019 and "other" on 2020 train on identical rows, and
their R coefficients are bit-identical, so this is one fitted model scored on
its two test subsets.

That makes the question specific: what is different about the model trained
on the 16,381-row set with fold 7 held out? This asks four things.

  1. Selected penalty. Does that fit sit at an unusual C, in either
     implementation, compared with its nine siblings?
  2. Sparsity. Does it keep an unusual number of features?
  3. Feature agreement. Do R and Python disagree about which features to
     keep more than they do elsewhere?
  4. The fold itself. Is anything odd about the held-out children, most
     obviously how many have the outcome at a 3% base rate.

Any of those could explain it as a property of that fold rather than a
defect in the port. None of them would justify moving the margin, which was
committed before the comparison ran.

Run from the repository root::

    uv run python analyses/diagnose_failing_split.py \\
        --r-coefficients      $COEF/NSCH_seed1_coefficients.csv \\
        --python-auc          analyses/glmnet_replication_lasso_seed1_is100.csv \\
        --r-predictions       $REF/NSCH_seed1_predictions_repaired.csv \\
        --python-predictions  $REF/python_lasso_seed1_is100_predictions.csv

where $COEF and $REF are the coefficients and predictions directories of the
reproduce-soak-nsch checkout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from scipy import stats

SPLIT_KEY = ["test_subset", "train_source", "fold"]
SPEARMAN_FLOOR = 0.95
# The two cells that share one fitted model, and the fold they hold out.
SUSPECT_FOLD = 7
SUSPECT_CELLS = (("2019", "same"), ("2020", "other"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r-coefficients", required=True)
    parser.add_argument("--python-auc", required=True)
    parser.add_argument("--r-predictions", required=True)
    parser.add_argument("--python-predictions", required=True)
    args = parser.parse_args()

    paths = {
        "R coefficients": Path(args.r_coefficients),
        "Python AUC": Path(args.python_auc),
        "R predictions": Path(args.r_predictions),
        "Python predictions": Path(args.python_predictions),
    }
    for label, path in paths.items():
        if not path.is_file():
            print(f"FAIL: {label} not found at {path}", file=sys.stderr)
            return 2

    r_coefficients = pl.read_csv(paths["R coefficients"]).with_columns(
        pl.col("test_subset").cast(pl.Utf8)
    )
    python_auc = (
        pl.read_csv(paths["Python AUC"])
        .filter(~pl.col("downsampled"))
        .with_columns(pl.col("test_subset").cast(pl.Utf8))
    )
    r_predictions = pl.read_csv(paths["R predictions"]).with_columns(
        pl.col("test_subset").cast(pl.Utf8)
    )
    python_predictions = pl.read_csv(paths["Python predictions"]).with_columns(
        pl.col("test_subset").cast(pl.Utf8)
    )

    # -------------------------------------------- 1. confirm one model
    print("=" * 88)
    print("1. Are the two failing cells really one fitted model?")
    print("=" * 88)
    first, second = SUSPECT_CELLS
    weights = {}
    for subset, source in SUSPECT_CELLS:
        cell = r_coefficients.filter(
            (pl.col("test_subset") == subset)
            & (pl.col("train_source") == source)
            & (pl.col("fold") == SUSPECT_FOLD)
        ).sort("feature")
        weights[(subset, source)] = cell["weight"].to_numpy()
    largest_gap = float(np.abs(weights[first] - weights[second]).max())
    print(
        f"  R coefficients, {first[0]} {first[1]} against {second[0]} {second[1]}, fold "
        f"{SUSPECT_FOLD}"
    )
    print(f"  largest absolute difference: {largest_gap:.3e}")
    print("  identical" if largest_gap == 0 else "  NOT identical, so the premise is wrong")

    # -------------------------------- 2. selected penalty and sparsity
    print("\n" + "=" * 88)
    print("2. Selected penalty and sparsity, this fold against its siblings")
    print("=" * 88)
    print(f"{'fold':>6}{'python C_1se':>14}{'R nonzero':>12}{'rho 1se':>10}")
    for fold in sorted(python_auc["fold"].unique().to_list()):
        auc_row = python_auc.filter(
            (pl.col("test_subset") == first[0])
            & (pl.col("train_source") == first[1])
            & (pl.col("fold") == fold)
        )
        coefficient_rows = r_coefficients.filter(
            (pl.col("test_subset") == first[0])
            & (pl.col("train_source") == first[1])
            & (pl.col("fold") == fold)
        )
        n_nonzero = int((coefficient_rows["weight"].to_numpy() != 0).sum())

        r_cell = r_predictions.filter(
            (pl.col("test_subset") == first[0])
            & (pl.col("train_source") == first[1])
            & (pl.col("fold") == fold)
        )
        python_cell = python_predictions.filter(
            (pl.col("test_subset") == first[0])
            & (pl.col("train_source") == first[1])
            & (pl.col("fold") == fold)
        )
        joined = r_cell.join(python_cell, on="row_id", how="inner", suffix="_py")
        rho = float(
            stats.spearmanr(joined["r_prob"].to_numpy(), joined["py_prob_1se"].to_numpy()).statistic
        )
        marker = "   <- fails" if rho < SPEARMAN_FLOOR else ""
        c_value = cast("float", auc_row["c_1se"][0]) if "c_1se" in auc_row.columns else float("nan")
        print(f"{fold:>6}{c_value:>14.5f}{n_nonzero:>12}{rho:>10.5f}{marker}")

    # ---------------------------------------- 3. feature disagreement
    print("\n" + "=" * 88)
    print("3. Where the two implementations rank children differently")
    print("=" * 88)
    r_cell = r_predictions.filter(
        (pl.col("test_subset") == first[0])
        & (pl.col("train_source") == first[1])
        & (pl.col("fold") == SUSPECT_FOLD)
    )
    python_cell = python_predictions.filter(
        (pl.col("test_subset") == first[0])
        & (pl.col("train_source") == first[1])
        & (pl.col("fold") == SUSPECT_FOLD)
    )
    joined = r_cell.join(python_cell, on="row_id", how="inner", suffix="_py")
    r_probability = joined["r_prob"].to_numpy()
    python_probability = joined["py_prob_1se"].to_numpy()
    rank_gap = np.abs(stats.rankdata(r_probability) - stats.rankdata(python_probability))
    print(f"  held-out children: {joined.height}")
    outcome_rate = cast("float", joined["truth"].mean())
    print(f"  outcome positives: {int(joined['truth'].sum())} ({outcome_rate:.4f})")
    print(
        f"  rank displacement: mean {rank_gap.mean():.1f}, "
        f"median {np.median(rank_gap):.1f}, worst {rank_gap.max():.0f}"
    )
    print(f"  children displaced by more than 100 ranks: {int((rank_gap > 100).sum())}")
    print(f"  probability MAD: {float(np.abs(python_probability - r_probability).mean()):.6f}")
    print("\n  The most displaced children, by rank:")
    worst_order = np.argsort(rank_gap)[::-1][:8]
    print(f"    {'row_id':>8}{'truth':>7}{'R prob':>10}{'Python prob':>13}{'rank gap':>10}")
    for index in worst_order:
        print(
            f"    {joined['row_id'][int(index)]:>8}{joined['truth'][int(index)]:>7}"
            f"{r_probability[index]:>10.5f}{python_probability[index]:>13.5f}"
            f"{rank_gap[index]:>10.0f}"
        )

    # ---------------------------------------------- 4. is the fold odd?
    print("\n" + "=" * 88)
    print("4. Is this fold unusual?")
    print("=" * 88)
    print(f"{'fold':>6}{'n test':>9}{'positives':>11}{'rate':>9}")
    for fold in sorted(r_predictions["fold"].unique().to_list()):
        cell = r_predictions.filter(
            (pl.col("test_subset") == first[0])
            & (pl.col("train_source") == first[1])
            & (pl.col("fold") == fold)
        )
        if cell.height == 0:
            continue
        positives = int(cell["truth"].sum())
        marker = "   <- fails" if fold == SUSPECT_FOLD else ""
        print(f"{fold:>6}{cell.height:>9}{positives:>11}{positives / cell.height:>9.4f}{marker}")

    print("\n" + "=" * 88)
    print("Reading this")
    print("=" * 88)
    print("  A fold with few positives gives Spearman little to work with: rank")
    print("  agreement among the 97% of children the model is confident about is")
    print("  driven by tiny probability differences, so a low correlation there")
    print("  need not mean the models disagree about anything that matters.")
    print()
    print("  If this fold looks ordinary on every count, the shortfall belongs to")
    print("  the port and is worth explaining rather than excusing. Either way the")
    print("  margin stands; it was fixed before the comparison ran.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
