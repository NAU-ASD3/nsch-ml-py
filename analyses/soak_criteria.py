"""How should we decide whether the Python port reproduces the R analysis?

The obvious answer is to run the SOAK t-test on both sides and check the
significance verdicts match. That turns out to be a poor criterion here,
and this script shows why using the numbers themselves.

Three criteria are compared:

  Verdict agreement   do R and Python both reject at alpha, in the same
                      direction? This is the intuitive choice.
  Estimate agreement  does each side's contrast estimate fall inside the
                      other side's confidence interval?
  Interval overlap    do the two intervals overlap at all? The weakest of
                      the three, included for contrast.

Verdict agreement is also recomputed under Bonferroni and Benjamini-
Hochberg, because with four tests here (and a couple of hundred once the
real analysis runs across outcomes, periods, learners, and fairness
variables) multiplicity is not a side issue.

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
SOURCES = ("same", "other", "all")
CONTRASTS = (("All - Same", "all", "same"), ("Other - Same", "other", "same"))


def paired(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = a - b
    n = len(d)
    mean = float(d.mean())
    sem = float(d.std(ddof=1)) / np.sqrt(n)
    t_stat, p = stats.ttest_rel(a, b)
    crit = stats.t.ppf(1 - ALPHA / 2, n - 1)
    return {
        "mean": mean,
        "sem": sem,
        "t": float(t_stat),
        "p": float(p),
        "lo": mean - crit * sem,
        "hi": mean + crit * sem,
    }


def bh(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, monotone."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = m - rank + 1
        val = min(prev, pvals[idx] * m / i)
        adj[idx] = val
        prev = val
    return adj.tolist()


def call(stat: dict[str, float], p: float) -> str:
    if p >= ALPHA:
        return "no difference"
    return "higher" if stat["mean"] > 0 else "lower"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="analyses/glmnet_replication_grid60.csv")
    ap.add_argument("--auc-col", default="auc_1se")
    args = ap.parse_args()

    path = Path(args.results)
    if not path.is_file():
        print(f"FAIL: {path} not found", file=sys.stderr)
        return 2
    df = (
        pl.read_csv(path)
        .filter(~pl.col("downsampled") & pl.col("r_auc").is_not_null())
        .with_columns(pl.col("test_subset").cast(pl.Utf8))
    )
    subsets = sorted(df["test_subset"].unique().to_list())
    print(f"{df.height} full splits, subsets {subsets}, Python column {args.auc_col}")

    def series(subset: str, source: str, col: str) -> np.ndarray:
        return (
            df.filter((pl.col("test_subset") == subset) & (pl.col("train_source") == source))
            .sort("fold")[col]
            .to_numpy()
            .astype(np.float64)
        )

    rows = []
    for name, a, b in CONTRASTS:
        for subset in subsets:
            rows.append(
                {
                    "contrast": name,
                    "subset": subset,
                    "R": paired(series(subset, a, "r_auc"), series(subset, b, "r_auc")),
                    "PY": paired(series(subset, a, args.auc_col), series(subset, b, args.auc_col)),
                }
            )

    r_raw = [row["R"]["p"] for row in rows]
    py_raw = [row["PY"]["p"] for row in rows]
    m = len(rows)
    r_bonf = [min(1.0, p * m) for p in r_raw]
    py_bonf = [min(1.0, p * m) for p in py_raw]
    r_bh, py_bh = bh(r_raw), bh(py_raw)

    # ---------------------------------------------------------------- table
    print("\n" + "=" * 88)
    print("Contrast estimates and intervals")
    print("=" * 88)
    print(
        f"{'contrast':<14}{'subset':>7}{'side':>8}{'estimate':>11}{'95% interval':>24}{'raw p':>9}"
    )
    for row in rows:
        for side, key in (("R", "R"), ("Python", "PY")):
            s = row[key]
            print(
                f"{row['contrast']:<14}{row['subset']:>7}{side:>8}{s['mean']:>+11.5f}"
                f"   [{s['lo']:+.5f}, {s['hi']:+.5f}]{s['p']:>9.4f}"
            )

    # ------------------------------------------------------- three criteria
    print("\n" + "=" * 88)
    print("Criterion 1: do the estimates agree?")
    print("=" * 88)
    print(
        f"{'contrast':<14}{'subset':>7}{'R in PY interval':>19}{'PY in R interval':>19}{'both':>8}"
    )
    est_ok = 0
    for row in rows:
        r, py = row["R"], row["PY"]
        a = py["lo"] <= r["mean"] <= py["hi"]
        b = r["lo"] <= py["mean"] <= r["hi"]
        est_ok += int(a and b)
        print(
            f"{row['contrast']:<14}{row['subset']:>7}{'yes' if a else 'NO':>19}"
            f"{'yes' if b else 'NO':>19}{'yes' if a and b else 'NO':>8}"
        )
    print(f"\n  agree: {est_ok} of {m}")

    print("\n" + "=" * 88)
    print("Criterion 2: do the intervals overlap?")
    print("=" * 88)
    ov_ok = 0
    for row in rows:
        r, py = row["R"], row["PY"]
        ok = not (r["hi"] < py["lo"] or py["hi"] < r["lo"])
        ov_ok += int(ok)
        print(f"{row['contrast']:<14}{row['subset']:>7}{'yes' if ok else 'NO':>10}")
    print(f"\n  agree: {ov_ok} of {m}")

    print("\n" + "=" * 88)
    print("Criterion 3: do the significance verdicts agree?")
    print("=" * 88)
    print(f"{'adjustment':<16}{'contrast':<14}{'subset':>7}{'R':>16}{'Python':>16}{'match':>8}")
    verdict_counts = {}
    for adj_name, r_ps, py_ps in (
        ("none", r_raw, py_raw),
        ("Bonferroni", r_bonf, py_bonf),
        ("Benjamini-H", r_bh, py_bh),
    ):
        n_ok = 0
        for row, rp, pp in zip(rows, r_ps, py_ps, strict=True):
            rc, pc = call(row["R"], rp), call(row["PY"], pp)
            ok = rc == pc
            n_ok += int(ok)
            print(
                f"{adj_name:<16}{row['contrast']:<14}{row['subset']:>7}{rc:>16}{pc:>16}"
                f"{'yes' if ok else 'NO':>8}"
            )
        verdict_counts[adj_name] = n_ok
        print(f"{'':<16}{'':<14}{'':>7}{'':>16}{'agree:':>16}{n_ok:>4} of {m}")
        print()

    # ------------------------------------------------------------- summary
    print("=" * 88)
    print("Summary")
    print("=" * 88)
    print(f"  estimates agree            {est_ok} of {m}")
    print(f"  intervals overlap          {ov_ok} of {m}")
    for k, v in verdict_counts.items():
        print(f"  verdicts agree ({k:<11}) {v} of {m}")

    spread = sorted(set(verdict_counts.values()))
    if len(spread) > 1:
        print()
        print("  Verdict agreement moves with the multiplicity adjustment, and the")
        print("  adjustment is a judgement call. A criterion that changes answer")
        print("  depending on a choice we make afterwards is not a good criterion.")
    if est_ok == m:
        print()
        print("  Estimate agreement holds throughout. Each side's contrast sits inside")
        print("  the other's interval in every comparison, which is the claim the")
        print("  replication can actually support.")

    contrast_scale = float(np.median([abs(row["R"]["mean"]) for row in rows]))
    impl_scale = float(
        np.median(
            [
                paired(series(s, src, args.auc_col), series(s, src, "r_auc"))["hi"]
                for s in subsets
                for src in SOURCES
            ]
        )
    )
    print()
    print(f"  typical contrast under test        {contrast_scale:.5f}")
    print(f"  typical implementation half-width  {impl_scale:.5f}")
    print(f"  ratio                              {impl_scale / contrast_scale:.1f}x")
    print()
    print("  At this ratio a dichotomous call sits close enough to the threshold")
    print("  that it flips on noise. Four tests here; the full analysis will run")
    print("  a couple of hundred across outcomes, periods, learners and fairness")
    print("  variables, so some verdicts will disagree no matter how good the port.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
