"""How much does the R analysis disagree with itself, and why?

The pairwise distances between R runs of the same analysis are not one
distribution. They fall into tight groups that agree to about 0.0005,
separated by gaps of around 0.007. Averaging across the gap gives a number
that describes neither, so this script finds the groups first and reports
within-group and between-group distances separately.

On the runs available as of 6 Aug 2026 the groups are:

  Feb/March runs (local, local_desktop, local_laptop, mpi, proj), sharing an
  older mlr3learners build.

  6 Aug runs (seed1, seed2, seed3, unseeded) on mlr3learners 0.14.0.9000,
  the tdhock/mlr3learners@cv_glmnet_seed fork.

  batchtools, on its own.

Two results come out of that. The cv.glmnet inner seed is worth about
0.0005, and leaving it unset gives bit-identical output to setting it to 1,
so the learner defaults to 1. The package version is worth about 0.0073,
which is what actually separates February from August.

Run from the repository root::

    uv run python analyses/r_vs_r.py \\
        --reproduce-dir ~/Documents/NAU/Grad/Research/ADSI/reproduce-soak-nsch
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

DEFAULT_REPRODUCE = Path.home() / "Documents/NAU/Grad/Research/ADSI/reproduce-soak-nsch"
ALPHA = 0.05
CONTRASTS = (("All - Same", "all", "same"), ("Other - Same", "other", "same"))
KEEP = ["test.subset", "train.subsets", "test.fold", "n.train.groups", "classif.auc"]


def load_run(path: Path) -> pl.DataFrame | None:
    """Read one R scores file, keeping the full split for each cell."""
    try:
        df = pl.read_csv(path, infer_schema_length=10000)
    except Exception as exc:
        print(f"  skip {path.name}: {exc}")
        return None
    if [c for c in KEEP if c not in df.columns]:
        return None
    if "learner_id" in df.columns:
        df = df.filter(pl.col("learner_id") == "classif.cv_glmnet")
    return (
        df.select(KEEP)
        .with_columns(pl.col("test.subset").cast(pl.Utf8))
        .sort("n.train.groups", descending=True)
        .unique(subset=["test.subset", "train.subsets", "test.fold"], keep="first")
        .sort(["test.subset", "train.subsets", "test.fold"])
    )


def aligned(a: pl.DataFrame, b: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    j = a.join(b, on=["test.subset", "train.subsets", "test.fold"], how="inner", suffix="_b")
    return (
        j["classif.auc"].to_numpy().astype(np.float64),
        j["classif.auc_b"].to_numpy().astype(np.float64),
    )


def find_clusters(names: list[str], dist: np.ndarray) -> dict[str, int]:
    """Single-linkage cut at the widest relative gap in the sorted distances.

    Bit-identical runs sit at distance zero. Including those would make the
    largest relative gap zero-to-anything and split every run into its own
    cluster, so the search for a cut runs over the positive distances only.
    Identical runs still end up grouped, because single linkage joins them
    below any positive threshold.
    """
    condensed = squareform(dist, checks=False)
    positive = np.sort(np.unique(condensed[condensed > 0]))
    n_zero = int(np.sum(condensed == 0))
    if n_zero:
        print(f"  {n_zero} pair(s) are bit-identical; excluded from the cut search")
    if len(positive) < 2:
        return dict.fromkeys(names, 1)
    ratios = positive[1:] / positive[:-1]
    i = int(np.argmax(ratios))
    threshold = float(np.sqrt(positive[i] * positive[i + 1]))
    print(
        f"  cutting between {positive[i]:.6f} and {positive[i + 1]:.6f} "
        f"({ratios[i]:.0f}x jump), threshold {threshold:.6f}"
    )
    labels = fcluster(linkage(condensed, method="single"), threshold, criterion="distance")
    return dict(zip(names, labels.tolist(), strict=True))


def contrast_stats(df: pl.DataFrame, subset: str, a: str, b: str) -> tuple[float, float]:
    """Mean difference and two-sided paired p-value for one SOAK contrast."""

    def pick(src: str) -> np.ndarray:
        return (
            df.filter((pl.col("test.subset") == subset) & (pl.col("train.subsets") == src))
            .sort("test.fold")["classif.auc"]
            .to_numpy()
            .astype(np.float64)
        )

    x, y = pick(a), pick(b)
    if len(x) != len(y) or len(x) < 2:
        return float("nan"), float("nan")
    p = float(stats.ttest_rel(x, y).pvalue)
    return float((x - y).mean()), p


def call(mean: float, p: float) -> str:
    if np.isnan(p):
        return "n/a"
    return "no difference" if p >= ALPHA else ("higher" if mean > 0 else "lower")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproduce-dir", default=str(DEFAULT_REPRODUCE))
    ap.add_argument("--python-results", default="analyses/glmnet_replication_grid60.csv")
    ap.add_argument("--auc-col", default="auc_1se")
    args = ap.parse_args()

    results = Path(os.environ.get("NSCH_REPRODUCE_DIR", args.reproduce_dir)) / "results"
    if not results.is_dir():
        print(f"FAIL: {results} not found", file=sys.stderr)
        return 2

    print("R result files")
    runs: dict[str, pl.DataFrame] = {}
    for path in sorted(results.rglob("*.csv")):
        if path.stem.startswith("meta") or "jobs" in path.stem:
            continue
        df = load_run(path)
        if df is None or df.height == 0:
            continue
        runs[f"{path.parent.name}/{path.stem}"] = df
        print(f"  {f'{path.parent.name}/{path.stem}':<34} {df.height:>4} splits")
    if len(runs) < 2:
        print("need at least two runs", file=sys.stderr)
        return 1

    names = sorted(runs)
    n = len(names)
    dist = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        x, y = aligned(runs[names[i]], runs[names[j]])
        d = float(np.abs(x - y).mean())
        dist[i, j] = dist[j, i] = d

    print("\n" + "=" * 78)
    print("Clustering the R runs by how far apart they are")
    print("=" * 78)
    labels = find_clusters(names, dist)
    groups: dict[int, list[str]] = {}
    for name, lab in labels.items():
        groups.setdefault(lab, []).append(name)
    for lab in sorted(groups):
        print(f"\n  cluster {lab}")
        for name in sorted(groups[lab]):
            print(f"    {name}")

    print("\n" + "=" * 78)
    print("Within clusters")
    print("=" * 78)
    within: dict[int, list[float]] = {}
    for lab, members in sorted(groups.items()):
        vals = [
            dist[names.index(a), names.index(b)]
            for a, b in itertools.combinations(sorted(members), 2)
        ]
        within[lab] = vals
        if vals:
            note = "  (bit-identical)" if max(vals) == 0 else ""
            print(
                f"  cluster {lab} ({len(members)} runs): mean {np.mean(vals):.6f}, "
                f"max {np.max(vals):.6f}{note}"
            )
        else:
            print(f"  cluster {lab} ({len(members)} run): single member")

    print("\n" + "=" * 78)
    print("Between clusters")
    print("=" * 78)
    for a, b in itertools.combinations(sorted(groups), 2):
        vals = [dist[names.index(x), names.index(y)] for x in groups[a] for y in groups[b]]
        print(f"  cluster {a} to cluster {b}: mean {np.mean(vals):.6f}")

    pooled = [v for vals in within.values() for v in vals]
    nonzero = [v for v in pooled if v > 0]
    floor = float(np.mean(nonzero)) if nonzero else 0.0
    if pooled:
        print(f"\n  within-cluster mean, all pairs:      {np.mean(pooled):.6f}")
        if nonzero:
            print(f"  within-cluster mean, non-identical:  {floor:.6f}")
        if len(nonzero) < len(pooled):
            print(f"  ({len(pooled) - len(nonzero)} identical pair(s) excluded from the floor)")

    # ------------------------------------------------------ verdicts
    print("\n" + "=" * 78)
    print("SOAK verdicts computed from each run")
    print("=" * 78)
    subsets = sorted(runs[names[0]]["test.subset"].unique().to_list())
    flips = 0
    for cname, a, b in CONTRASTS:
        for subset in subsets:
            print(f"\n  {cname}, test subset {subset}")
            calls = set()
            for name in names:
                mean, p = contrast_stats(runs[name], subset, a, b)
                v = call(mean, p)
                calls.add(v)
                marker = " <- within 0.01 of alpha" if abs(p - ALPHA) < 0.01 else ""
                print(f"    [{labels[name]}] {name:<34}{mean:>+10.5f}{p:>9.4f}  {v}{marker}")
            if len(calls) > 1:
                flips += 1
                print(f"    -> RUNS DISAGREE: {sorted(calls)}")
            else:
                print("    -> all runs agree")
    print(
        f"\n  contrasts where R runs disagree with each other: {flips} of "
        f"{len(CONTRASTS) * len(subsets)}"
    )

    # ------------------------------------------------------ Python
    py_path = Path(args.python_results)
    if py_path.is_file():
        py = (
            pl.read_csv(py_path)
            .filter(~pl.col("downsampled") & pl.col("r_auc").is_not_null())
            .with_columns(pl.col("test_subset").cast(pl.Utf8))
            .rename(
                {"test_subset": "test.subset", "train_source": "train.subsets", "fold": "test.fold"}
            )
        )
        py_frame = py.select(
            ["test.subset", "train.subsets", "test.fold", pl.col(args.auc_col).alias("classif.auc")]
        )
        print("\n" + "=" * 78)
        print("Where the Python port sits")
        print("=" * 78)
        per_cluster: dict[int, float] = {}
        for lab in sorted(groups):
            vals = []
            for name in groups[lab]:
                x, y = aligned(py_frame, runs[name])
                vals.append(float(np.abs(x - y).mean()))
            per_cluster[lab] = float(np.mean(vals))
            print(f"  to cluster {lab} ({len(groups[lab])} runs): mean {per_cluster[lab]:.6f}")

        nearest_lab = min(per_cluster, key=lambda k: per_cluster[k])
        nearest = per_cluster[nearest_lab]
        print(f"\n  nearest cluster: {nearest_lab}, at {nearest:.6f}")
        if floor > 0:
            print(f"  within-cluster floor: {floor:.6f}")
            print(f"  the port is {nearest / floor:.0f}x the floor")
        else:
            print("  every within-cluster pair is identical, so there is no floor to")
            print("  divide by; compare the figure above against the between-cluster")
            print("  distances instead.")
        print()
        print("  The comparison worth making is against the between-cluster distances.")
        print("  If the port's nearest distance is well below those, it sits closer to")
        print("  the current reference than two versions of the reference sit to each")
        print("  other, and the remaining gap is implementation rather than build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
