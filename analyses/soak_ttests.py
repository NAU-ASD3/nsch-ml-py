"""SOAK t-tests and figures, in the form Hocking et al. use.

The SOAK paper compares training subsets with a two-sided paired t-test
on K-1 = 9 degrees of freedom, computed separately for each test subset
over the 10 cross-validation folds, reporting the mean difference and a
p-value (Section 3, Figures 5-7). This applies that test three ways and
draws the two summary figures.

Part A - Did we reach the same scientific conclusion?
    All-Same and Other-Same for each test subset, once on the R results
    and once on ours, checking whether the two agree in sign and in
    significance verdict. This is the replication test that matters.

Part B - The published reference.
    The paper reports mean AUC on NSCH_autism test subset 2020 of 0.9670
    for All and 0.9658 for Same, with accuracy 0.9759 for both.

Part C - R versus Python, per subset and source.
    Paired t9 on the AUC difference, with a confidence interval. A large
    p-value here is not evidence of equivalence, only absence of evidence
    of difference; with 10 folds the test has little power. The interval
    against a stated margin is the defensible summary.

Part D - Two figures, after Figures 4 and 6/7 of the paper.

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

PAPER_2020 = {"all_auc": 0.9670, "same_auc": 0.9658, "acc": 0.9759}
ALPHA = 0.05
MARGIN = 0.01
SOURCES = ("same", "other", "all")
R_COLOR = "#B85042"
PY_COLOR = "#003466"


def paired_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float, float]:
    """Two-sided paired t-test on a - b. Returns mean, t, p, and CI bounds."""
    d = a - b
    n = len(d)
    mean = float(d.mean())
    if n < 2 or np.allclose(d, d[0]):
        return mean, float("nan"), float("nan"), mean, mean
    t_stat, p = stats.ttest_rel(a, b)
    sem = float(d.std(ddof=1)) / np.sqrt(n)
    crit = stats.t.ppf(1 - ALPHA / 2, n - 1)
    return mean, float(t_stat), float(p), mean - crit * sem, mean + crit * sem


def verdict(mean: float, p: float) -> str:
    if np.isnan(p):
        return "identical"
    if p >= ALPHA:
        return "no difference"
    return "higher" if mean > 0 else "lower"


def make_figures(df: pl.DataFrame, subsets: list[str], auc_col: str, outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(subset: str, source: str, col: str) -> np.ndarray:
        sel = df.filter(
            (pl.col("test_subset") == subset) & (pl.col("train_source") == source)
        ).sort("fold")
        return sel[col].to_numpy().astype(np.float64)

    outdir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: mean +/- sd AUC by train source, after paper Figure 4 ---
    fig, axes = plt.subplots(1, len(subsets), figsize=(3.6 * len(subsets), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, subset in zip(axes, subsets, strict=True):
        for offset, (label, col, color) in enumerate(
            (("R", "r_auc", R_COLOR), ("Python", auc_col, PY_COLOR))
        ):
            means = [float(series(subset, s, col).mean()) for s in SOURCES]
            sds = [float(series(subset, s, col).std(ddof=1)) for s in SOURCES]
            ypos = np.arange(len(SOURCES)) + (offset - 0.5) * 0.22
            ax.errorbar(
                means,
                ypos,
                xerr=sds,
                fmt="o",
                color=color,
                capsize=3,
                markersize=5,
                label=label,
                linewidth=1.4,
            )
        ax.set_yticks(np.arange(len(SOURCES)))
        ax.set_yticklabels([s.capitalize() for s in SOURCES])
        ax.set_title(f"test subset {subset}", fontsize=10)
        ax.set_xlabel("AUC")
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        ax.invert_yaxis()
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Mean and standard deviation of test AUC over 10 folds", fontsize=11)
    fig.tight_layout()
    p1 = outdir / "soak_auc_by_source.png"
    fig.savefig(p1, dpi=170)
    plt.close(fig)

    # --- Figure 2: contrast differences with CI and p-values, after 6 and 7 ---
    contrasts = (("All - Same", "all", "same"), ("Other - Same", "other", "same"))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharex=True)
    for ax, (name, src_a, src_b) in zip(axes, contrasts, strict=True):
        labels, ypos = [], []
        row = 0
        for subset in subsets:
            for label, col, color in (("R", "r_auc", R_COLOR), ("Python", auc_col, PY_COLOR)):
                mean, _, p, lo, hi = paired_t(
                    series(subset, src_a, col), series(subset, src_b, col)
                )
                ax.errorbar(
                    [mean],
                    [row],
                    xerr=[[mean - lo], [hi - mean]],
                    fmt="o",
                    color=color,
                    capsize=3,
                    markersize=5,
                    linewidth=1.4,
                )
                star = " *" if (not np.isnan(p) and p < ALPHA) else ""
                ax.annotate(
                    f"p={p:.3f}{star}",
                    (hi, row),
                    textcoords="offset points",
                    xytext=(6, 0),
                    va="center",
                    fontsize=8,
                    color=color,
                )
                labels.append(f"{subset} {label}")
                ypos.append(row)
                row += 1
            row += 0.6
        ax.axvline(0, color="#777777", linewidth=1, linestyle="--")
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("difference in AUC")
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        ax.invert_yaxis()
        ax.margins(x=0.25)
    fig.suptitle("SOAK contrasts with 95% intervals and paired t9 p-values", fontsize=11)
    fig.tight_layout()
    p2 = outdir / "soak_contrasts.png"
    fig.savefig(p2, dpi=170)
    plt.close(fig)

    print(f"\nwrote {p1}")
    print(f"wrote {p2}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="analyses/glmnet_replication.csv")
    ap.add_argument("--auc-col", default="auc_min")
    ap.add_argument("--figdir", default="analyses/figures")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    path = Path(args.results)
    if not path.is_file():
        print(f"FAIL: {path} not found; run run_glmnet_replication.py first", file=sys.stderr)
        return 2

    # test_subset round-trips through CSV as an integer, so normalise to text.
    df = (
        pl.read_csv(path)
        .filter(~pl.col("downsampled") & pl.col("r_auc").is_not_null())
        .with_columns(pl.col("test_subset").cast(pl.Utf8))
    )
    print(f"loaded {df.height} full splits from {path}")
    subsets = sorted(df["test_subset"].unique().to_list())
    print(f"test subsets: {subsets}")

    def series(subset: str, source: str, col: str) -> np.ndarray:
        sel = df.filter(
            (pl.col("test_subset") == subset) & (pl.col("train_source") == source)
        ).sort("fold")
        return sel[col].to_numpy().astype(np.float64)

    # ---------------- Part A ----------------
    print("\n" + "=" * 78)
    print("A.  SOAK comparisons, two-sided paired t on 9 df, per test subset")
    print("=" * 78)
    agree = disagree = 0
    for contrast, src_a, src_b in (
        ("All - Same", "all", "same"),
        ("Other - Same", "other", "same"),
    ):
        print(f"\n  {contrast}")
        print(
            f"    {'subset':>8}  {'source':>8}  {'mean diff':>10}  {'t':>7}  "
            f"{'p':>9}  {'verdict':>14}"
        )
        for subset in subsets:
            calls = {}
            for label, col in (("R", "r_auc"), ("Python", args.auc_col)):
                mean, t_stat, p, _, _ = paired_t(
                    series(subset, src_a, col), series(subset, src_b, col)
                )
                calls[label] = (mean, verdict(mean, p))
                print(
                    f"    {subset:>8}  {label:>8}  {mean:+10.5f}  {t_stat:7.2f}  "
                    f"{p:9.4f}  {verdict(mean, p):>14}"
                )
            same_sign = np.sign(calls["R"][0]) == np.sign(calls["Python"][0])
            same_call = calls["R"][1] == calls["Python"][1]
            if same_sign and same_call:
                agree += 1
                print(f"    {'':>8}  {'':>8}  -> same conclusion")
            else:
                disagree += 1
                note = "same sign, different significance call" if same_sign else "opposite signs"
                print(f"    {'':>8}  {'':>8}  -> CONCLUSIONS DIFFER ({note})")
    print(f"\n  Conclusions agreeing: {agree} of {agree + disagree}")

    # ---------------- Part B ----------------
    print("\n" + "=" * 78)
    print("B.  Against the published NSCH_autism numbers (2020 test subset)")
    print("=" * 78)
    if "2020" in subsets:
        for label, src, key in (("All", "all", "all_auc"), ("Same", "same", "same_auc")):
            for who, col in (("R", "r_auc"), ("Python", args.auc_col)):
                ours = float(series("2020", src, col).mean())
                theirs = PAPER_2020[key]
                print(
                    f"    train on {label:<5} {who:>7}   mean AUC {ours:.4f}   "
                    f"paper {theirs:.4f}   diff {ours - theirs:+.4f}"
                )
        print("\n    The paper rounds to four places, so agreement near 0.001 is the")
        print("    most this comparison can demonstrate.")
    else:
        print(f"    test subset 2020 not found; have {subsets}")

    # ---------------- Part C ----------------
    print("\n" + "=" * 78)
    print("C.  R versus Python, same test rows, paired t on 9 df")
    print("=" * 78)
    print(
        f"    {'subset':>8}  {'source':>7}  {'mean diff':>10}  {'t':>7}  {'p':>9}  "
        f"{'95% CI':>22}  {'within':>8}"
    )
    inside = total = 0
    spreads = []
    for subset in subsets:
        for source in SOURCES:
            mean, t_stat, p, lo, hi = paired_t(
                series(subset, source, args.auc_col), series(subset, source, "r_auc")
            )
            ok = abs(lo) < MARGIN and abs(hi) < MARGIN
            inside += int(ok)
            total += 1
            spreads.append(max(abs(lo), abs(hi)))
            print(
                f"    {subset:>8}  {source:>7}  {mean:+10.5f}  {t_stat:7.2f}  {p:9.4f}  "
                f"[{lo:+.5f},{hi:+.5f}]  {'yes' if ok else 'NO':>8}"
            )
    print(f"\n    Intervals inside +/-{MARGIN}: {inside} of {total}")

    # ---------------- The scale problem ----------------
    print("\n" + "=" * 78)
    print("D.  How the two scales compare")
    print("=" * 78)
    contrast_sizes = []
    for subset in subsets:
        for src_a, src_b in (("all", "same"), ("other", "same")):
            for col in ("r_auc", args.auc_col):
                m, _, _, _, _ = paired_t(series(subset, src_a, col), series(subset, src_b, col))
                contrast_sizes.append(abs(m))
    print(f"    Typical SOAK contrast being tested : {np.median(contrast_sizes):.5f}")
    print(f"    Typical R-vs-Python interval half-width : {np.median(spreads):.5f}")
    ratio = np.median(spreads) / max(np.median(contrast_sizes), 1e-12)
    print(f"    Ratio : {ratio:.1f}x")
    print()
    print("    This is the number that matters for setting a tolerance. When the")
    print("    implementation gap is the same size as the effect SOAK is testing,")
    print("    a replication can pass an absolute-difference check and still flip")
    print("    a published significance call. A tolerance has to be small relative")
    print("    to the contrasts, not merely small in AUC units.")

    if not args.no_figures:
        make_figures(df, subsets, args.auc_col, Path(args.figdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
