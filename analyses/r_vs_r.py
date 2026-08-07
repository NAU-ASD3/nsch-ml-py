"""How much does the R analysis disagree with itself, and why?

The pairwise distances between R runs of the same analysis are not one
distribution. They fall into tight groups whose members agree closely,
separated by much larger gaps. Averaging across a gap gives a number that
describes neither side of it, so this script finds the groups first and
reports within-group and between-group distances separately, then places
the Python port against each group.

Findings are recorded in ``docs/replication-equivalence.md``, which is the
single source for the numbers. They are deliberately not restated here,
because figures embedded in a docstring go stale as runs are added.

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
EXPECTED_SPLITS = 60
CONTRASTS = (("All - Same", "all", "same"), ("Other - Same", "other", "same"))
SPLIT_KEY = ["test.subset", "train.subsets", "test.fold"]
REQUIRED_COLUMNS = [*SPLIT_KEY, "n.train.groups", "classif.auc"]


def load_run(path: Path) -> pl.DataFrame | None:
    """Read one R scores file, keeping the full split for each cell.

    Returns None, with a reason on stdout, when the file cannot be read or
    does not look like a scores file. Silence would be the wrong default for
    a script whose whole job is scanning a directory.
    """
    try:
        scores = pl.read_csv(path, infer_schema_length=10000)
    except Exception as error:
        print(f"  skip {path.name}: unreadable ({error})")
        return None
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in scores.columns]
    if missing_columns:
        print(f"  skip {path.name}: missing column(s) {missing_columns}")
        return None
    if "learner_id" in scores.columns:
        scores = scores.filter(pl.col("learner_id") == "classif.cv_glmnet")
    if scores.height == 0:
        print(f"  skip {path.name}: no classif.cv_glmnet rows")
        return None
    return (
        scores.select(REQUIRED_COLUMNS)
        .with_columns(pl.col("test.subset").cast(pl.Utf8))
        # Downsampled variants of a cell have a smaller train size. Keeping the
        # largest gives the full split, which is what the reference files hold.
        .sort("n.train.groups", descending=True)
        .unique(subset=SPLIT_KEY, keep="first")
        .sort(SPLIT_KEY)
    )


def aligned_auc(left: pl.DataFrame, right: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return both AUC vectors over the splits present in both frames.

    An inner join drops anything that does not match, so a partial results
    file would quietly yield a distance computed from a handful of rows.
    Callers should check the returned length against EXPECTED_SPLITS.
    """
    joined = left.join(right, on=SPLIT_KEY, how="inner", suffix="_right")
    return (
        joined["classif.auc"].to_numpy().astype(np.float64),
        joined["classif.auc_right"].to_numpy().astype(np.float64),
    )


def mean_absolute_distance(left: pl.DataFrame, right: pl.DataFrame) -> tuple[float, int]:
    """Mean absolute AUC difference between two runs, and how many splits it used."""
    left_auc, right_auc = aligned_auc(left, right)
    return float(np.abs(left_auc - right_auc).mean()), len(left_auc)


def find_clusters(run_names: list[str], distances: np.ndarray) -> dict[str, int]:
    """Group runs by single linkage, cut at the widest relative gap.

    Bit-identical runs sit at distance zero. Including those would make the
    largest relative gap zero-to-anything and split every run into its own
    cluster, so the search for a cut runs over the positive distances only.
    Identical runs still end up grouped, because single linkage joins them
    below any positive threshold.

    This assumes the distances really are clustered. The chosen gap is
    printed so a smooth distribution, where the cut would be arbitrary, is
    visible rather than hidden.
    """
    condensed = squareform(distances, checks=False)
    positive = np.sort(np.unique(condensed[condensed > 0]))
    n_identical = int(np.sum(condensed == 0))
    if n_identical:
        print(f"  {n_identical} pair(s) are bit-identical; excluded from the cut search")
    if len(positive) < 2:
        return dict.fromkeys(run_names, 1)
    gap_ratios = positive[1:] / positive[:-1]
    widest = int(np.argmax(gap_ratios))
    threshold = float(np.sqrt(positive[widest] * positive[widest + 1]))  # geometric midpoint
    print(
        f"  cutting between {positive[widest]:.6f} and {positive[widest + 1]:.6f} "
        f"({gap_ratios[widest]:.0f}x jump), threshold {threshold:.6f}"
    )
    cluster_ids = fcluster(linkage(condensed, method="single"), threshold, criterion="distance")
    return dict(zip(run_names, cluster_ids.tolist(), strict=True))


def contrast_stats(
    run: pl.DataFrame, test_subset: str, source_a: str, source_b: str
) -> tuple[float, float]:
    """Mean difference and two-sided paired p-value for one SOAK contrast."""

    def fold_series(train_source: str) -> np.ndarray:
        return (
            run.filter(
                (pl.col("test.subset") == test_subset) & (pl.col("train.subsets") == train_source)
            )
            .sort("test.fold")["classif.auc"]
            .to_numpy()
            .astype(np.float64)
        )

    auc_a, auc_b = fold_series(source_a), fold_series(source_b)
    if len(auc_a) != len(auc_b) or len(auc_a) < 2:
        return float("nan"), float("nan")
    return float((auc_a - auc_b).mean()), float(stats.ttest_rel(auc_a, auc_b).pvalue)


def significance_call(mean_diff: float, p_value: float) -> str:
    """Verdict at ALPHA, or 'n/a' when the test could not run."""
    if np.isnan(p_value):
        return "n/a"
    if p_value >= ALPHA:
        return "no difference"
    return "higher" if mean_diff > 0 else "lower"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reproduce-dir", default=str(DEFAULT_REPRODUCE))
    # Defaults match the analysis documented in docs/replication-equivalence.md.
    parser.add_argument("--python-results", default="analyses/glmnet_replication_grid60.csv")
    parser.add_argument("--auc-col", default="auc_1se")
    args = parser.parse_args()

    results_dir = Path(os.environ.get("NSCH_REPRODUCE_DIR", args.reproduce_dir)) / "results"
    if not results_dir.is_dir():
        print(f"FAIL: {results_dir} not found", file=sys.stderr)
        return 2

    print("R result files")
    runs: dict[str, pl.DataFrame] = {}
    for path in sorted(results_dir.rglob("*.csv")):
        if path.stem.startswith("meta") or "jobs" in path.stem:
            continue
        run = load_run(path)
        if run is None:
            continue
        run_name = f"{path.parent.name}/{path.stem}"
        runs[run_name] = run
        flag = "" if run.height == EXPECTED_SPLITS else f"  <- expected {EXPECTED_SPLITS}"
        print(f"  {run_name:<34} {run.height:>4} splits{flag}")
    if len(runs) < 2:
        print("need at least two readable runs", file=sys.stderr)
        return 1

    run_names = sorted(runs)
    n_runs = len(run_names)
    distances = np.zeros((n_runs, n_runs))
    n_partial_pairs = 0
    for index_a, index_b in itertools.combinations(range(n_runs), 2):
        name_a, name_b = run_names[index_a], run_names[index_b]
        distance, n_shared = mean_absolute_distance(runs[name_a], runs[name_b])
        if n_shared != EXPECTED_SPLITS:
            n_partial_pairs += 1
            print(
                f"  WARNING: {name_a} and {name_b} share only {n_shared} splits; "
                f"their distance is computed from those"
            )
        distances[index_a, index_b] = distances[index_b, index_a] = distance
    if n_partial_pairs:
        print(f"\n  {n_partial_pairs} pair(s) did not align on all {EXPECTED_SPLITS} splits.")
        print("  Treat the distances below as provisional until that is resolved.")

    print("\n" + "=" * 78)
    print("Clustering the R runs by how far apart they are")
    print("=" * 78)
    cluster_of = find_clusters(run_names, distances)
    members_of: dict[int, list[str]] = {}
    for run_name, cluster_id in cluster_of.items():
        members_of.setdefault(cluster_id, []).append(run_name)
    for cluster_id in sorted(members_of):
        print(f"\n  cluster {cluster_id}")
        for run_name in sorted(members_of[cluster_id]):
            print(f"    {run_name}")

    print("\n" + "=" * 78)
    print("Within clusters")
    print("=" * 78)
    within_cluster: dict[int, list[float]] = {}
    for cluster_id, members in sorted(members_of.items()):
        pair_distances = [
            distances[run_names.index(member_a), run_names.index(member_b)]
            for member_a, member_b in itertools.combinations(sorted(members), 2)
        ]
        within_cluster[cluster_id] = pair_distances
        if pair_distances:
            note = "  (bit-identical)" if max(pair_distances) == 0 else ""
            print(
                f"  cluster {cluster_id} ({len(members)} runs): "
                f"mean {np.mean(pair_distances):.6f}, max {np.max(pair_distances):.6f}{note}"
            )
        else:
            print(f"  cluster {cluster_id} ({len(members)} run): single member")

    print("\n" + "=" * 78)
    print("Between clusters")
    print("=" * 78)
    for cluster_a, cluster_b in itertools.combinations(sorted(members_of), 2):
        cross_distances = [
            distances[run_names.index(member_a), run_names.index(member_b)]
            for member_a in members_of[cluster_a]
            for member_b in members_of[cluster_b]
        ]
        print(f"  cluster {cluster_a} to cluster {cluster_b}: mean {np.mean(cross_distances):.6f}")

    all_within = [value for values in within_cluster.values() for value in values]
    # Identical pairs contribute a zero that drags the mean down and makes any
    # ratio against it look better than it is, so the floor excludes them.
    non_identical = [value for value in all_within if value > 0]
    floor = float(np.mean(non_identical)) if non_identical else 0.0
    if all_within:
        print(f"\n  within-cluster mean, all pairs:      {np.mean(all_within):.6f}")
        if non_identical:
            print(f"  within-cluster mean, non-identical:  {floor:.6f}")
        if len(non_identical) < len(all_within):
            n_excluded = len(all_within) - len(non_identical)
            print(f"  ({n_excluded} identical pair(s) excluded from the floor)")

    # ------------------------------------------------------ verdicts
    print("\n" + "=" * 78)
    print("SOAK verdicts computed from each run")
    print("=" * 78)
    test_subsets = sorted(runs[run_names[0]]["test.subset"].unique().to_list())
    n_disagreements = 0
    for contrast_name, source_a, source_b in CONTRASTS:
        for test_subset in test_subsets:
            print(f"\n  {contrast_name}, test subset {test_subset}")
            verdicts = set()
            for run_name in run_names:
                mean_diff, p_value = contrast_stats(runs[run_name], test_subset, source_a, source_b)
                verdict = significance_call(mean_diff, p_value)
                verdicts.add(verdict)
                marker = ""
                if not np.isnan(p_value) and abs(p_value - ALPHA) < 0.01:
                    marker = " <- within 0.01 of alpha"
                print(
                    f"    [{cluster_of[run_name]}] {run_name:<34}"
                    f"{mean_diff:>+10.5f}{p_value:>9.4f}  {verdict}{marker}"
                )
            # "n/a" means a run could not be tested, which is a data problem
            # rather than a disagreement, so report it separately.
            if "n/a" in verdicts:
                print("    -> at least one run could not be tested; check the file")
                verdicts.discard("n/a")
            if len(verdicts) > 1:
                n_disagreements += 1
                print(f"    -> RUNS DISAGREE: {sorted(verdicts)}")
            elif verdicts:
                print("    -> all runs agree")
    print(
        f"\n  contrasts where R runs disagree with each other: {n_disagreements} of "
        f"{len(CONTRASTS) * len(test_subsets)}"
    )

    # ------------------------------------------------------ Python
    python_path = Path(args.python_results)
    if not python_path.is_file():
        print(f"\n(skipping the Python comparison: {python_path} not found)")
        return 0

    python_splits = (
        pl.read_csv(python_path)
        .filter(~pl.col("downsampled") & pl.col("r_auc").is_not_null())
        .with_columns(pl.col("test_subset").cast(pl.Utf8))
        .rename(
            {
                "test_subset": "test.subset",
                "train_source": "train.subsets",
                "fold": "test.fold",
            }
        )
    )
    if args.auc_col not in python_splits.columns:
        print(f"FAIL: column {args.auc_col!r} not in {python_path}", file=sys.stderr)
        return 2
    python_run = python_splits.select([*SPLIT_KEY, pl.col(args.auc_col).alias("classif.auc")])

    print("\n" + "=" * 78)
    print("Where the Python port sits")
    print("=" * 78)
    distance_to_cluster: dict[int, float] = {}
    for cluster_id in sorted(members_of):
        cluster_distances = []
        for run_name in members_of[cluster_id]:
            distance, n_shared = mean_absolute_distance(python_run, runs[run_name])
            if n_shared != EXPECTED_SPLITS:
                print(f"  WARNING: Python and {run_name} share only {n_shared} splits")
            cluster_distances.append(distance)
        distance_to_cluster[cluster_id] = float(np.mean(cluster_distances))
        print(
            f"  to cluster {cluster_id} ({len(members_of[cluster_id])} runs): "
            f"mean {distance_to_cluster[cluster_id]:.6f}"
        )

    nearest_cluster = min(distance_to_cluster, key=lambda key: distance_to_cluster[key])
    nearest_distance = distance_to_cluster[nearest_cluster]
    print(f"\n  nearest cluster: {nearest_cluster}, at {nearest_distance:.6f}")
    if floor > 0:
        print(f"  within-cluster floor: {floor:.6f}")
        print(f"  the port is {nearest_distance / floor:.0f}x the floor")
    else:
        print("  every within-cluster pair is identical, so there is no floor to")
        print("  divide by; compare the figure above against the between-cluster")
        print("  distances instead.")
    print()
    print("  The comparison worth making is against the between-cluster distances.")
    print("  If the port's nearest distance is well below those, it sits closer to")
    print("  the current reference than two builds of the reference sit to each")
    print("  other, and the remaining gap is implementation rather than build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
