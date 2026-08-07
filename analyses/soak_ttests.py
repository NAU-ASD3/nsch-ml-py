"""SOAK t-tests and figures, in the form Hocking et al. use.

The SOAK paper compares training subsets with a two-sided paired t-test on
K-1 = 9 degrees of freedom, computed separately for each test subset over
the 10 cross-validation folds, reporting the mean difference and a p-value
(Section 3, Figures 5-7). This applies that test three ways and draws two
summary figures.

Part A - Did we reach the same scientific conclusion?
    All-Same and Other-Same for each test subset, once on the R results and
    once on ours, checking whether the two agree in sign and in verdict.

Part B - The published reference.
    The paper reports mean AUC on NSCH_autism test subset 2020 of 0.9670
    training on All and 0.9658 training on Same.

Part C - R versus Python, per subset and source.
    Paired t9 on the AUC difference, with a confidence interval. A large
    p-value here is not evidence of equivalence, only absence of evidence of
    difference; with 10 folds the test has little power.

Part D - How the size of the implementation gap compares to the size of the
    contrasts being tested.

Findings are recorded in ``docs/replication-equivalence.md``, which is the
single source for the numbers. This module deliberately does not restate
them, because figures embedded in a docstring go stale.

Run from the repository root, after run_glmnet_replication.py::

    uv run --with matplotlib python analyses/soak_ttests.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

# Hocking et al. 2026, Section 4.3, NSCH_autism test subset 2020.
PAPER_2020 = {"all_auc": 0.9670, "same_auc": 0.9658}
ALPHA = 0.05

# Illustrative only. Part C prints how many intervals fall inside this band so
# the magnitude is visible, but an absolute AUC margin is NOT the replication
# criterion: see Part D for why, and soak_criteria.py for what is proposed.
ILLUSTRATIVE_MARGIN = 0.01

EXPECTED_FOLDS = 10
TRAIN_SOURCES = ("same", "other", "all")
CONTRASTS = (("All - Same", "all", "same"), ("Other - Same", "other", "same"))
R_COLOR = "#B85042"
PYTHON_COLOR = "#003466"


class PairedResult(dict[str, float]):
    """One paired t-test: mean, t, p, and the interval bounds lo and hi."""


def paired_test(first: np.ndarray, second: np.ndarray) -> PairedResult:
    """Two-sided paired t-test on ``first - second``.

    Degenerate input (fewer than two pairs, or a constant difference) yields
    nan for the statistic and p-value and a zero-width interval at the mean,
    rather than raising. A malformed input file should surface as a visible
    nan in the table, not as a stack trace partway through it.
    """
    diffs = first - second
    n_pairs = len(diffs)
    if n_pairs < 2 or np.allclose(diffs, diffs[0]):
        mean_diff = float(diffs.mean()) if n_pairs else float("nan")
        return PairedResult(
            mean=mean_diff, t=float("nan"), p=float("nan"), lo=mean_diff, hi=mean_diff
        )
    mean_diff = float(diffs.mean())
    standard_error = float(diffs.std(ddof=1)) / np.sqrt(n_pairs)
    critical_value = float(stats.t.ppf(1 - ALPHA / 2, n_pairs - 1))
    test_result = stats.ttest_rel(first, second)
    return PairedResult(
        mean=mean_diff,
        t=float(test_result.statistic),
        p=float(test_result.pvalue),
        lo=mean_diff - critical_value * standard_error,
        hi=mean_diff + critical_value * standard_error,
    )


def significance_call(mean_diff: float, p_value: float) -> str:
    """Verdict at ALPHA, or 'n/a' when the test could not run."""
    if np.isnan(p_value):
        return "n/a"
    if p_value >= ALPHA:
        return "no difference"
    return "higher" if mean_diff > 0 else "lower"


def fold_series(
    splits: pl.DataFrame, test_subset: str, train_source: str, column: str
) -> np.ndarray:
    """Per-fold values for one (test subset, train source) cell, in fold order."""
    return (
        splits.filter(
            (pl.col("test_subset") == test_subset) & (pl.col("train_source") == train_source)
        )
        .sort("fold")[column]
        .to_numpy()
        .astype(np.float64)
    )


def draw_auc_figure(
    splits: pl.DataFrame, test_subsets: list[str], python_column: str, output_dir: Path
) -> Path:
    """Mean and standard deviation of AUC by train source, after paper Figure 4."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, len(test_subsets), figsize=(3.6 * len(test_subsets), 3.6), sharey=True
    )
    axes = np.atleast_1d(axes)
    for axis, test_subset in zip(axes, test_subsets, strict=True):
        for offset, (side_label, column, color) in enumerate(
            (("R", "r_auc", R_COLOR), ("Python", python_column, PYTHON_COLOR))
        ):
            means = [
                float(fold_series(splits, test_subset, source, column).mean())
                for source in TRAIN_SOURCES
            ]
            deviations = [
                float(fold_series(splits, test_subset, source, column).std(ddof=1))
                for source in TRAIN_SOURCES
            ]
            # Nudge the two series apart vertically so markers do not overlap.
            marker_y = np.arange(len(TRAIN_SOURCES)) + (offset - 0.5) * 0.22
            axis.errorbar(
                means,
                marker_y,
                xerr=deviations,
                fmt="o",
                color=color,
                capsize=3,
                markersize=5,
                label=side_label,
                linewidth=1.4,
            )
        axis.set_yticks(np.arange(len(TRAIN_SOURCES)))
        axis.set_yticklabels([source.capitalize() for source in TRAIN_SOURCES])
        axis.set_title(f"test subset {test_subset}", fontsize=10)
        axis.set_xlabel("AUC")
        axis.grid(axis="x", alpha=0.25, linewidth=0.6)
        axis.invert_yaxis()
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    figure.suptitle("Mean and standard deviation of test AUC over 10 folds", fontsize=11)
    figure.tight_layout()
    path = output_dir / "soak_auc_by_source.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def draw_contrast_figure(
    splits: pl.DataFrame, test_subsets: list[str], python_column: str, output_dir: Path
) -> Path:
    """Contrast differences with intervals and p-values, after paper Figures 6-7."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(CONTRASTS), figsize=(10.5, 3.4), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, (contrast_name, source_a, source_b) in zip(axes, CONTRASTS, strict=True):
        row_labels: list[str] = []
        row_positions: list[float] = []
        row = 0.0
        for test_subset in test_subsets:
            for side_label, column, color in (
                ("R", "r_auc", R_COLOR),
                ("Python", python_column, PYTHON_COLOR),
            ):
                contrast = paired_test(
                    fold_series(splits, test_subset, source_a, column),
                    fold_series(splits, test_subset, source_b, column),
                )
                axis.errorbar(
                    [contrast["mean"]],
                    [row],
                    xerr=[
                        [contrast["mean"] - contrast["lo"]],
                        [contrast["hi"] - contrast["mean"]],
                    ],
                    fmt="o",
                    color=color,
                    capsize=3,
                    markersize=5,
                    linewidth=1.4,
                )
                star = " *" if (not np.isnan(contrast["p"]) and contrast["p"] < ALPHA) else ""
                axis.annotate(
                    f"p={contrast['p']:.3f}{star}",
                    (contrast["hi"], row),
                    textcoords="offset points",
                    xytext=(6, 0),
                    va="center",
                    fontsize=8,
                    color=color,
                )
                row_labels.append(f"{test_subset} {side_label}")
                row_positions.append(row)
                row += 1
            # Gap between test subsets.
            row += 0.6
        axis.axvline(0, color="#777777", linewidth=1, linestyle="--")
        axis.set_yticks(row_positions)
        axis.set_yticklabels(row_labels, fontsize=9)
        axis.set_title(contrast_name, fontsize=10)
        axis.set_xlabel("difference in AUC")
        axis.grid(axis="x", alpha=0.25, linewidth=0.6)
        axis.invert_yaxis()
        # Leave room on the right for the p-value annotations.
        axis.margins(x=0.25)
    figure.suptitle("SOAK contrasts with 95% intervals and paired t9 p-values", fontsize=11)
    figure.tight_layout()
    path = output_dir / "soak_contrasts.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def make_figures(
    splits: pl.DataFrame, test_subsets: list[str], python_column: str, output_dir: Path
) -> None:
    """Write both summary figures to output_dir."""
    import matplotlib

    matplotlib.use("Agg")

    output_dir.mkdir(parents=True, exist_ok=True)
    auc_figure = draw_auc_figure(splits, test_subsets, python_column, output_dir)
    contrast_figure = draw_contrast_figure(splits, test_subsets, python_column, output_dir)
    print(f"\nwrote {auc_figure}")
    print(f"wrote {contrast_figure}")


def report_part_a(splits: pl.DataFrame, test_subsets: list[str], python_column: str) -> None:
    """Do R and Python reach the same conclusion on each SOAK contrast?"""
    print("\n" + "=" * 78)
    print("A.  SOAK comparisons, two-sided paired t on 9 df, per test subset")
    print("=" * 78)
    n_agree = n_disagree = 0
    for contrast_name, source_a, source_b in CONTRASTS:
        print(f"\n  {contrast_name}")
        print(
            f"    {'subset':>8}  {'side':>8}  {'mean diff':>10}  {'t':>7}  "
            f"{'p':>9}  {'verdict':>14}"
        )
        for test_subset in test_subsets:
            sides: dict[str, tuple[float, str]] = {}
            for side_label, column in (("R", "r_auc"), ("Python", python_column)):
                contrast = paired_test(
                    fold_series(splits, test_subset, source_a, column),
                    fold_series(splits, test_subset, source_b, column),
                )
                verdict = significance_call(contrast["mean"], contrast["p"])
                sides[side_label] = (contrast["mean"], verdict)
                print(
                    f"    {test_subset:>8}  {side_label:>8}  {contrast['mean']:+10.5f}  "
                    f"{contrast['t']:7.2f}  {contrast['p']:9.4f}  {verdict:>14}"
                )
            same_sign = np.sign(sides["R"][0]) == np.sign(sides["Python"][0])
            same_verdict = sides["R"][1] == sides["Python"][1]
            if same_sign and same_verdict:
                n_agree += 1
                print(f"    {'':>8}  {'':>8}  -> same conclusion")
            else:
                n_disagree += 1
                reason = "same sign, different significance call" if same_sign else "opposite signs"
                print(f"    {'':>8}  {'':>8}  -> CONCLUSIONS DIFFER ({reason})")
    print(f"\n  Conclusions agreeing: {n_agree} of {n_agree + n_disagree}")


def report_part_b(splits: pl.DataFrame, test_subsets: list[str], python_column: str) -> None:
    """Compare our 2020 means against the figures printed in the paper."""
    print("\n" + "=" * 78)
    print("B.  Against the published NSCH_autism numbers (2020 test subset)")
    print("=" * 78)
    if "2020" not in test_subsets:
        print(f"    test subset 2020 not found; have {test_subsets}")
        return
    for source_label, train_source, paper_key in (
        ("All", "all", "all_auc"),
        ("Same", "same", "same_auc"),
    ):
        for side_label, column in (("R", "r_auc"), ("Python", python_column)):
            ours = float(fold_series(splits, "2020", train_source, column).mean())
            published = PAPER_2020[paper_key]
            print(
                f"    train on {source_label:<5} {side_label:>7}   mean AUC {ours:.4f}   "
                f"paper {published:.4f}   diff {ours - published:+.4f}"
            )
    print("\n    The paper rounds to four places, so agreement near 0.001 is the")
    print("    most this comparison can demonstrate.")


def report_parts_c_and_d(splits: pl.DataFrame, test_subsets: list[str], python_column: str) -> None:
    """The implementation gap, and how it compares to the contrasts being tested."""
    print("\n" + "=" * 78)
    print("C.  R versus Python, same test rows, paired t on 9 df")
    print("=" * 78)
    print(
        f"    {'subset':>8}  {'source':>7}  {'mean diff':>10}  {'t':>7}  {'p':>9}  "
        f"{'95% CI':>22}  {'within':>8}"
    )
    n_inside = n_total = 0
    half_widths = []
    for test_subset in test_subsets:
        for train_source in TRAIN_SOURCES:
            gap = paired_test(
                fold_series(splits, test_subset, train_source, python_column),
                fold_series(splits, test_subset, train_source, "r_auc"),
            )
            inside = abs(gap["lo"]) < ILLUSTRATIVE_MARGIN and abs(gap["hi"]) < ILLUSTRATIVE_MARGIN
            n_inside += int(inside)
            n_total += 1
            half_widths.append((gap["hi"] - gap["lo"]) / 2)
            print(
                f"    {test_subset:>8}  {train_source:>7}  {gap['mean']:+10.5f}  "
                f"{gap['t']:7.2f}  {gap['p']:9.4f}  [{gap['lo']:+.5f},{gap['hi']:+.5f}]  "
                f"{'yes' if inside else 'NO':>8}"
            )
    print(f"\n    Intervals inside +/-{ILLUSTRATIVE_MARGIN}: {n_inside} of {n_total}")
    print("    That band is illustrative, not the criterion. See Part D.")

    print("\n" + "=" * 78)
    print("D.  How the two scales compare")
    print("=" * 78)
    contrast_sizes = [
        abs(
            paired_test(
                fold_series(splits, test_subset, source_a, column),
                fold_series(splits, test_subset, source_b, column),
            )["mean"]
        )
        for test_subset in test_subsets
        for _, source_a, source_b in CONTRASTS
        for column in ("r_auc", python_column)
    ]
    typical_contrast = float(np.median(contrast_sizes))
    typical_half_width = float(np.median(half_widths))
    print(f"    Typical SOAK contrast being tested      : {typical_contrast:.5f}")
    print(f"    Typical R-vs-Python interval half-width : {typical_half_width:.5f}")
    if typical_contrast > 0:
        print(
            f"    Ratio                                   : "
            f"{typical_half_width / typical_contrast:.1f}x"
        )
    print()
    print("    This is the number that matters for setting a tolerance. When the")
    print("    implementation gap is several times the effect SOAK is testing, a")
    print("    replication can pass an absolute-difference check and still flip a")
    print("    published significance call. A tolerance has to be small relative to")
    print("    the contrasts, not merely small in AUC units.")


def main() -> int:
    parser = argparse.ArgumentParser()
    # Defaults match the analysis documented in docs/replication-equivalence.md.
    # Changing them without updating that document will make the two disagree.
    parser.add_argument("--results", default="analyses/glmnet_replication_grid60.csv")
    parser.add_argument("--auc-col", default="auc_1se")
    parser.add_argument("--figdir", default="analyses/figures")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.is_file():
        print(
            f"FAIL: {results_path} not found; run run_glmnet_replication.py first",
            file=sys.stderr,
        )
        return 2

    # test_subset round-trips through CSV as an integer, so normalise to text.
    splits = (
        pl.read_csv(results_path)
        .filter(~pl.col("downsampled") & pl.col("r_auc").is_not_null())
        .with_columns(pl.col("test_subset").cast(pl.Utf8))
    )
    print(f"loaded {splits.height} full splits from {results_path}")
    if args.auc_col not in splits.columns:
        print(f"FAIL: column {args.auc_col!r} not in {results_path}", file=sys.stderr)
        return 2
    test_subsets = sorted(splits["test_subset"].unique().to_list())
    print(f"test subsets: {test_subsets}   Python column: {args.auc_col}")

    # A short series means the results file is incomplete. Comparing a partial
    # run against a complete one looks like real disagreement, so say so.
    incomplete_cells = [
        (test_subset, train_source, len(fold_series(splits, test_subset, train_source, "r_auc")))
        for test_subset in test_subsets
        for train_source in TRAIN_SOURCES
        if len(fold_series(splits, test_subset, train_source, "r_auc")) != EXPECTED_FOLDS
    ]
    if incomplete_cells:
        print(f"\nWARNING: {len(incomplete_cells)} cell(s) do not have {EXPECTED_FOLDS} folds:")
        for test_subset, train_source, n_found in incomplete_cells:
            print(f"  {test_subset} {train_source}: {n_found}")

    report_part_a(splits, test_subsets, args.auc_col)
    report_part_b(splits, test_subsets, args.auc_col)
    report_parts_c_and_d(splits, test_subsets, args.auc_col)

    if not args.no_figures:
        make_figures(splits, test_subsets, args.auc_col, Path(args.figdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
