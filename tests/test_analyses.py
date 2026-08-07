"""Tests for the pure helpers in ``analyses/``.

These scripts are run by hand rather than imported by the package, but two of
their functions feed claims in ``docs/replication-equivalence.md``: the
Benjamini-Hochberg adjustment behind the verdict counts, and the clustering
that separates the R runs by build. A silent error in either would change a
stated result without failing anything, so both are covered here.

``analyses/`` is not a package, so the scripts are loaded by path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

if TYPE_CHECKING:
    from types import ModuleType

ANALYSES_DIR = Path(__file__).resolve().parent.parent / "analyses"


def load_script(name: str) -> ModuleType:
    """Import a script from analyses/ by path."""
    spec = importlib.util.spec_from_file_location(f"analyses_{name}", ANALYSES_DIR / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery
        pytest.skip(f"cannot load analyses/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


criteria = load_script("soak_criteria")
r_vs_r = load_script("r_vs_r")


# --- benjamini_hochberg ---------------------------------------------------


def test_bh_matches_worked_example() -> None:
    """Hand-computed step-up on four p-values.

    Sorted they are 0.01, 0.02, 0.03, 0.04 with four tests. Scaling each by
    n_tests / rank gives 0.04 four times, and the running minimum taken from
    the largest downward leaves all four at 0.04.
    """
    adjusted = criteria.benjamini_hochberg([0.01, 0.02, 0.03, 0.04])
    assert adjusted == pytest.approx([0.04, 0.04, 0.04, 0.04])


def test_bh_is_monotone_in_the_input() -> None:
    """A larger raw p-value can never receive a smaller adjusted value."""
    raw_p_values = [0.001, 0.2, 0.02, 0.9, 0.045]
    adjusted = criteria.benjamini_hochberg(raw_p_values)
    ascending = np.argsort(raw_p_values)
    adjusted_in_rank_order = [adjusted[index] for index in ascending]
    assert adjusted_in_rank_order == sorted(adjusted_in_rank_order)


def test_bh_never_exceeds_one() -> None:
    adjusted = criteria.benjamini_hochberg([0.5, 0.6, 0.7, 0.99])
    assert all(value <= 1.0 for value in adjusted)


def test_bh_leaves_a_single_p_value_alone() -> None:
    assert criteria.benjamini_hochberg([0.03]) == pytest.approx([0.03])


def test_bh_handles_an_empty_list() -> None:
    assert criteria.benjamini_hochberg([]) == []


def test_bh_is_at_least_as_large_as_the_raw_value() -> None:
    raw_p_values = [0.001, 0.01, 0.04, 0.2]
    adjusted = criteria.benjamini_hochberg(raw_p_values)
    assert all(
        adjusted_value >= raw_value
        for adjusted_value, raw_value in zip(adjusted, raw_p_values, strict=True)
    )


# --- paired_test ----------------------------------------------------------


def test_paired_test_recovers_a_known_mean() -> None:
    first = np.array([1.0, 2.0, 3.0, 4.0])
    second = np.array([0.5, 1.5, 2.5, 3.5])
    result = criteria.paired_test(first, second)
    assert result["mean"] == pytest.approx(0.5)


def test_paired_test_returns_nan_on_a_constant_difference() -> None:
    """Zero variance would make the t statistic infinite; nan is the signal."""
    first = np.array([1.0, 2.0, 3.0])
    second = np.array([0.0, 1.0, 2.0])
    result = criteria.paired_test(first, second)
    assert result["mean"] == pytest.approx(1.0)
    assert np.isnan(result["p"])
    assert result["lo"] == result["hi"] == pytest.approx(1.0)


def test_paired_test_returns_nan_on_a_single_pair() -> None:
    result = criteria.paired_test(np.array([1.0]), np.array([0.0]))
    assert np.isnan(result["p"])


def test_paired_test_interval_brackets_the_mean() -> None:
    generator = np.random.default_rng(0)
    first = generator.normal(size=20)
    second = first + generator.normal(scale=0.1, size=20)
    result = criteria.paired_test(first, second)
    assert result["lo"] < result["mean"] < result["hi"]


# --- significance_call ----------------------------------------------------


def test_significance_call_reports_direction_when_significant() -> None:
    assert criteria.significance_call(0.5, 0.01) == "higher"
    assert criteria.significance_call(-0.5, 0.01) == "lower"


def test_significance_call_reports_no_difference_at_and_above_alpha() -> None:
    assert criteria.significance_call(0.5, 0.05) == "no difference"
    assert criteria.significance_call(0.5, 0.9) == "no difference"


def test_significance_call_reports_na_for_an_untestable_comparison() -> None:
    assert criteria.significance_call(0.5, float("nan")) == "n/a"


# --- find_clusters --------------------------------------------------------


def square_matrix(pair_distances: dict[tuple[int, int], float], size: int) -> np.ndarray:
    """Build a symmetric distance matrix from the upper-triangle entries."""
    matrix = np.zeros((size, size))
    for (row, column), distance in pair_distances.items():
        matrix[row, column] = matrix[column, row] = distance
    return matrix


def test_find_clusters_splits_two_tight_groups() -> None:
    """Two pairs that agree closely, far apart from each other."""
    run_names = ["a", "b", "c", "d"]
    distances = square_matrix(
        {
            (0, 1): 0.0005,
            (2, 3): 0.0005,
            (0, 2): 0.007,
            (0, 3): 0.007,
            (1, 2): 0.007,
            (1, 3): 0.007,
        },
        4,
    )
    cluster_of = r_vs_r.find_clusters(run_names, distances)
    assert cluster_of["a"] == cluster_of["b"]
    assert cluster_of["c"] == cluster_of["d"]
    assert cluster_of["a"] != cluster_of["c"]


def test_find_clusters_groups_identical_runs_without_shattering() -> None:
    """A zero distance must not drive the cut.

    Before this was handled, an identical pair made the largest relative gap
    zero-to-anything and every run landed in its own cluster.
    """
    run_names = ["a", "b", "c", "d"]
    distances = square_matrix(
        {
            (0, 1): 0.0,
            (0, 2): 0.0005,
            (1, 2): 0.0005,
            (0, 3): 0.007,
            (1, 3): 0.007,
            (2, 3): 0.007,
        },
        4,
    )
    cluster_of = r_vs_r.find_clusters(run_names, distances)
    assert cluster_of["a"] == cluster_of["b"] == cluster_of["c"]
    assert cluster_of["d"] != cluster_of["a"]


def test_find_clusters_returns_one_group_when_nothing_separates() -> None:
    run_names = ["a", "b", "c"]
    distances = square_matrix({(0, 1): 0.001, (0, 2): 0.001, (1, 2): 0.001}, 3)
    cluster_of = r_vs_r.find_clusters(run_names, distances)
    assert len(set(cluster_of.values())) == 1


def test_find_clusters_labels_every_run() -> None:
    run_names = ["a", "b", "c", "d"]
    distances = square_matrix(
        {
            (0, 1): 0.0005,
            (2, 3): 0.0005,
            (0, 2): 0.007,
            (0, 3): 0.007,
            (1, 2): 0.007,
            (1, 3): 0.007,
        },
        4,
    )
    assert set(r_vs_r.find_clusters(run_names, distances)) == set(run_names)


# --- contrast_stats -------------------------------------------------------


def test_contrast_stats_returns_nan_on_mismatched_lengths() -> None:
    """An incomplete cell must not be silently compared against a full one."""
    run = pl.DataFrame(
        {
            "test.subset": ["2019"] * 5,
            "train.subsets": ["all"] * 3 + ["same"] * 2,
            "test.fold": [1, 2, 3, 1, 2],
            "classif.auc": [0.9, 0.91, 0.92, 0.88, 0.89],
        }
    )
    mean_diff, p_value = r_vs_r.contrast_stats(run, "2019", "all", "same")
    assert np.isnan(mean_diff)
    assert np.isnan(p_value)
