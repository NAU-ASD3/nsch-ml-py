"""How should we decide whether the Python port reproduces the R analysis?

The obvious answer is to run the SOAK t-test on both sides and check that
the significance verdicts match. That turns out to be a poor criterion, and
this script shows why using the numbers themselves.

Three criteria are compared:

  Verdict agreement   do R and Python both reject at alpha, in the same
                      direction? The intuitive choice.
  Estimate agreement  does each side's contrast estimate fall inside the
                      other side's confidence interval?
  Interval overlap    do the two intervals overlap at all? The weakest of
                      the three, included for contrast.

Verdict agreement is also recomputed under Bonferroni and Benjamini-
Hochberg. With a handful of tests here, and a couple of hundred once the
real analysis runs across outcomes, periods, learners and fairness
variables, multiplicity is not a side issue.

Findings are recorded in ``docs/replication-equivalence.md``, which is the
single source for the numbers.

Run from the repository root::

    uv run python analyses/soak_criteria.py \\
        --results analyses/glmnet_replication_grid60.csv --auc-col auc_1se
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ALPHA = 0.05
EXPECTED_FOLDS = 10
TRAIN_SOURCES = ("same", "other", "all")
CONTRASTS = (("All - Same", "all", "same"), ("Other - Same", "other", "same"))


def paired_test(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Two-sided paired t-test on ``first - second``, with its interval.

    Degenerate input (fewer than two pairs, or a constant difference) yields
    nan for the statistic and p-value and a zero-width interval, rather than
    raising. A malformed results file should show up as a visible nan in the
    table, not as a stack trace partway through.
    """
    diffs = first - second
    n_pairs = len(diffs)
    if n_pairs < 2 or np.allclose(diffs, diffs[0]):
        mean_diff = float(diffs.mean()) if n_pairs else float("nan")
        return {
            "mean": mean_diff,
            "sem": 0.0,
            "t": float("nan"),
            "p": float("nan"),
            "lo": mean_diff,
            "hi": mean_diff,
        }
    mean_diff = float(diffs.mean())
    standard_error = float(diffs.std(ddof=1)) / np.sqrt(n_pairs)
    critical_value = float(stats.t.ppf(1 - ALPHA / 2, n_pairs - 1))
    test_result = stats.ttest_rel(first, second)
    return {
        "mean": mean_diff,
        "sem": standard_error,
        "t": float(test_result.statistic),
        "p": float(test_result.pvalue),
        "lo": mean_diff - critical_value * standard_error,
        "hi": mean_diff + critical_value * standard_error,
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values.

    Step-up procedure: sort ascending, scale each by ``n_tests / rank``, then
    walk down from the largest taking a running minimum so the result stays
    monotone in the original p-values.
    """
    n_tests = len(p_values)
    if n_tests == 0:
        return []
    ascending = np.argsort(p_values).tolist()
    adjusted = np.empty(n_tests, dtype=float)
    running_min = 1.0
    for position, original_index in enumerate(reversed(ascending), start=1):
        rank = n_tests - position + 1
        running_min = min(running_min, p_values[original_index] * n_tests / rank)
        adjusted[original_index] = running_min
    return adjusted.tolist()


def significance_call(mean_diff: float, p_value: float) -> str:
    """Verdict at ALPHA, or 'n/a' when the test could not run."""
    if np.isnan(p_value):
        return "n/a"
    if p_value >= ALPHA:
        return "no difference"
    return "higher" if mean_diff > 0 else "lower"


def main() -> int:
    parser = argparse.ArgumentParser()
    # Defaults match the analysis documented in docs/replication-equivalence.md.
    parser.add_argument("--results", default="analyses/glmnet_replication_grid60.csv")
    parser.add_argument("--auc-col", default="auc_1se")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.is_file():
        print(f"FAIL: {results_path} not found", file=sys.stderr)
        return 2
    splits = (
        pl.read_csv(results_path)
        .filter(~pl.col("downsampled") & pl.col("r_auc").is_not_null())
        .with_columns(pl.col("test_subset").cast(pl.Utf8))
    )
    if args.auc_col not in splits.columns:
        print(f"FAIL: column {args.auc_col!r} not in {results_path}", file=sys.stderr)
        return 2
    test_subsets = sorted(splits["test_subset"].unique().to_list())
    print(f"{splits.height} full splits, subsets {test_subsets}, Python column {args.auc_col}")

    def fold_series(test_subset: str, train_source: str, column: str) -> np.ndarray:
        """The ten per-fold values for one (subset, source) cell."""
        return (
            splits.filter(
                (pl.col("test_subset") == test_subset) & (pl.col("train_source") == train_source)
            )
            .sort("fold")[column]
            .to_numpy()
            .astype(np.float64)
        )

    incomplete_cells = [
        (test_subset, train_source, len(fold_series(test_subset, train_source, "r_auc")))
        for test_subset in test_subsets
        for train_source in TRAIN_SOURCES
        if len(fold_series(test_subset, train_source, "r_auc")) != EXPECTED_FOLDS
    ]
    if incomplete_cells:
        print(f"\nWARNING: {len(incomplete_cells)} cell(s) do not have {EXPECTED_FOLDS} folds:")
        for test_subset, train_source, n_found in incomplete_cells:
            print(f"  {test_subset} {train_source}: {n_found}")

    comparisons = []
    for contrast_name, source_a, source_b in CONTRASTS:
        for test_subset in test_subsets:
            comparisons.append(
                {
                    "contrast": contrast_name,
                    "subset": test_subset,
                    "r": paired_test(
                        fold_series(test_subset, source_a, "r_auc"),
                        fold_series(test_subset, source_b, "r_auc"),
                    ),
                    "python": paired_test(
                        fold_series(test_subset, source_a, args.auc_col),
                        fold_series(test_subset, source_b, args.auc_col),
                    ),
                }
            )

    n_tests = len(comparisons)
    r_raw = [row["r"]["p"] for row in comparisons]
    python_raw = [row["python"]["p"] for row in comparisons]
    r_bonferroni = [min(1.0, p_value * n_tests) for p_value in r_raw]
    python_bonferroni = [min(1.0, p_value * n_tests) for p_value in python_raw]
    r_bh = benjamini_hochberg(r_raw)
    python_bh = benjamini_hochberg(python_raw)

    # ---------------------------------------------------------------- table
    print("\n" + "=" * 88)
    print("Contrast estimates and intervals")
    print("=" * 88)
    print(
        f"{'contrast':<14}{'subset':>7}{'side':>8}{'estimate':>11}{'95% interval':>24}{'raw p':>9}"
    )
    for row in comparisons:
        for side_label, side_key in (("R", "r"), ("Python", "python")):
            stat = row[side_key]
            print(
                f"{row['contrast']:<14}{row['subset']:>7}{side_label:>8}{stat['mean']:>+11.5f}"
                f"   [{stat['lo']:+.5f}, {stat['hi']:+.5f}]{stat['p']:>9.4f}"
            )

    # ------------------------------------------------------- three criteria
    print("\n" + "=" * 88)
    print("Criterion 1: do the estimates agree?")
    print("=" * 88)
    print(
        f"{'contrast':<14}{'subset':>7}{'R in PY interval':>19}{'PY in R interval':>19}{'both':>8}"
    )
    n_estimates_agree = 0
    for row in comparisons:
        r_stat, python_stat = row["r"], row["python"]
        r_inside_python = python_stat["lo"] <= r_stat["mean"] <= python_stat["hi"]
        python_inside_r = r_stat["lo"] <= python_stat["mean"] <= r_stat["hi"]
        both = r_inside_python and python_inside_r
        n_estimates_agree += int(both)
        print(
            f"{row['contrast']:<14}{row['subset']:>7}{'yes' if r_inside_python else 'NO':>19}"
            f"{'yes' if python_inside_r else 'NO':>19}{'yes' if both else 'NO':>8}"
        )
    print(f"\n  agree: {n_estimates_agree} of {n_tests}")

    print("\n" + "=" * 88)
    print("Criterion 2: do the intervals overlap?")
    print("=" * 88)
    n_intervals_overlap = 0
    for row in comparisons:
        r_stat, python_stat = row["r"], row["python"]
        overlaps = not (r_stat["hi"] < python_stat["lo"] or python_stat["hi"] < r_stat["lo"])
        n_intervals_overlap += int(overlaps)
        print(f"{row['contrast']:<14}{row['subset']:>7}{'yes' if overlaps else 'NO':>10}")
    print(f"\n  agree: {n_intervals_overlap} of {n_tests}")

    print("\n" + "=" * 88)
    print("Criterion 3: do the significance verdicts agree?")
    print("=" * 88)
    print(f"{'adjustment':<16}{'contrast':<14}{'subset':>7}{'R':>16}{'Python':>16}{'match':>8}")
    verdict_agreement: dict[str, int] = {}
    for adjustment_name, r_adjusted, python_adjusted in (
        ("none", r_raw, python_raw),
        ("Bonferroni", r_bonferroni, python_bonferroni),
        ("Benjamini-H", r_bh, python_bh),
    ):
        n_matching = 0
        for row, r_p, python_p in zip(comparisons, r_adjusted, python_adjusted, strict=True):
            r_verdict = significance_call(row["r"]["mean"], r_p)
            python_verdict = significance_call(row["python"]["mean"], python_p)
            matches = r_verdict == python_verdict
            n_matching += int(matches)
            print(
                f"{adjustment_name:<16}{row['contrast']:<14}{row['subset']:>7}"
                f"{r_verdict:>16}{python_verdict:>16}{'yes' if matches else 'NO':>8}"
            )
        verdict_agreement[adjustment_name] = n_matching
        print(f"{'':<16}{'':<14}{'':>7}{'':>16}{'agree:':>16}{n_matching:>4} of {n_tests}")
        print()

    # ------------------------------------------------------------- summary
    print("=" * 88)
    print("Summary")
    print("=" * 88)
    print(f"  estimates agree            {n_estimates_agree} of {n_tests}")
    print(f"  intervals overlap          {n_intervals_overlap} of {n_tests}")
    for adjustment_name, n_matching in verdict_agreement.items():
        print(f"  verdicts agree ({adjustment_name:<11}) {n_matching} of {n_tests}")

    if len(set(verdict_agreement.values())) > 1:
        print()
        print("  Verdict agreement moves with the multiplicity adjustment, and the")
        print("  adjustment is a judgement call. A criterion that changes answer")
        print("  depending on a choice we make afterwards is not a good criterion.")
    if n_estimates_agree == n_tests:
        print()
        print("  Estimate agreement holds throughout. Each side's contrast sits inside")
        print("  the other's interval in every comparison, which is the claim the")
        print("  replication can actually support.")

    # Scale check. The half-width is (hi - lo) / 2, not the upper bound: those
    # coincide only when the mean sits at zero, and relying on that would hide
    # a systematic offset in some future learner.
    # Median over both implementations, matching soak_ttests.py. Taking R
    # alone would answer a narrower question and give a different ratio.
    typical_contrast = float(
        np.median([abs(row[side]["mean"]) for row in comparisons for side in ("r", "python")])
    )
    implementation_gaps = [
        paired_test(
            fold_series(test_subset, train_source, args.auc_col),
            fold_series(test_subset, train_source, "r_auc"),
        )
        for test_subset in test_subsets
        for train_source in TRAIN_SOURCES
    ]
    typical_half_width = float(
        np.median([(gap["hi"] - gap["lo"]) / 2 for gap in implementation_gaps])
    )
    print()
    print(f"  typical contrast under test        {typical_contrast:.5f}")
    print(f"  typical implementation half-width  {typical_half_width:.5f}")
    if typical_contrast > 0:
        print(f"  ratio                              {typical_half_width / typical_contrast:.1f}x")
    print()
    print("  At this ratio a dichotomous call sits close enough to the threshold")
    print("  that it flips on noise. A handful of tests here; the full analysis")
    print("  will run a couple of hundred across outcomes, periods, learners and")
    print("  fairness variables, so some verdicts will disagree no matter how")
    print("  good the port is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
