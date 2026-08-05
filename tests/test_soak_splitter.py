"""Tests for the SOAK splitter.

Synthetic data throughout: three subsets, forty rows, a 25% outcome so
stratification is observable. The set-identity tests recompute expected
index sets from the SOAK definitions with plain boolean masks, so the
implementation is checked against first principles rather than itself.
The sizes=0 tests encode the rule read from ResamplingSameOtherSizesCV:
nominal per-subset train sizes same=floor(full*(K-1)/K), all=sum(same),
other=all-same; target = per-test-subset minimum of those; downsampled
per-(subset, outcome)-stratum counts floor(L_s * target / nominal_own).
"""

from collections.abc import Iterable

import numpy as np
import polars as pl
import pytest

from nsch_ml.soak import SoakSplit, TrainSource, assign_folds, ignore_group_kfold, iter_soak_splits

N_FOLDS = 4
SEED = 17


def make_data() -> tuple[np.ndarray, np.ndarray]:
    """Three subsets (16 + 12 + 12 rows), 10 positives of 40, scattered."""
    subset = np.array(["P1"] * 16 + ["P2"] * 12 + ["P3"] * 12)
    outcome = np.zeros(40, dtype=np.int64)
    outcome[[0, 5, 9, 14, 17, 20, 25, 29, 33, 38]] = 1  # 4 in P1, 3 in P2, 3 in P3
    return subset, outcome


def nominal_counts(subset: np.ndarray, n_folds: int) -> dict[str, dict[str, int]]:
    """The R rule: same=floor(full*(K-1)/K) per subset; all=sum(same); other=all-same."""
    labels = np.unique(subset)
    same = {s: int(np.sum(subset == s) * (n_folds - 1) // n_folds) for s in labels}
    all_count = sum(same.values())
    return {s: {"same": same[s], "other": all_count - same[s], "all": all_count} for s in labels}


# --------------------------------------------------------------------------
# assign_folds
# --------------------------------------------------------------------------


def test_assign_folds_returns_aligned_int64_in_range() -> None:
    subset, outcome = make_data()
    folds = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    assert folds.shape == subset.shape
    assert folds.dtype == np.int64
    assert np.all((folds >= 1) & (folds <= N_FOLDS))


def test_assign_folds_stratifies_every_cell_within_one() -> None:
    subset, outcome = make_data()
    folds = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    for s in np.unique(subset):
        for y in (0, 1):
            cell = folds[(subset == s) & (outcome == y)]
            counts = np.bincount(cell, minlength=N_FOLDS + 1)[1:]
            assert counts.max() - counts.min() <= 1


def test_assign_folds_deterministic_given_seed() -> None:
    subset, outcome = make_data()
    a = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    b = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    assert np.array_equal(a, b)


def test_assign_folds_changes_with_seed() -> None:
    subset, outcome = make_data()
    a = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    b = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED + 1)
    assert not np.array_equal(a, b)


def test_assign_folds_accepts_polars_series() -> None:
    subset, outcome = make_data()
    from_np = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    from_pl = assign_folds(pl.Series(subset), pl.Series(outcome), n_folds=N_FOLDS, seed=SEED)
    assert np.array_equal(from_np, from_pl)


def test_assign_folds_precomputed_passes_through_unchanged() -> None:
    subset, outcome = make_data()
    given = np.tile(np.arange(1, N_FOLDS + 1), 10)
    out = assign_folds(subset, outcome, n_folds=N_FOLDS, precomputed=given)
    assert np.array_equal(out, given)
    assert out.dtype == np.int64


def test_assign_folds_precomputed_rejects_out_of_range() -> None:
    subset, outcome = make_data()
    bad = np.ones(40, dtype=np.int64)
    bad[0] = N_FOLDS + 1
    with pytest.raises(ValueError):
        assign_folds(subset, outcome, n_folds=N_FOLDS, precomputed=bad)


def test_assign_folds_precomputed_rejects_non_integer() -> None:
    subset, outcome = make_data()
    bad = np.full(40, 1.5)
    with pytest.raises(ValueError):
        assign_folds(subset, outcome, n_folds=N_FOLDS, precomputed=bad)


def test_assign_folds_rejects_mismatched_lengths() -> None:
    subset, outcome = make_data()
    with pytest.raises(ValueError):
        assign_folds(subset[:-1], outcome, n_folds=N_FOLDS, seed=SEED)


def test_assign_folds_rejects_too_few_folds() -> None:
    subset, outcome = make_data()
    with pytest.raises(ValueError):
        assign_folds(subset, outcome, n_folds=1, seed=SEED)


def test_assign_folds_requires_seed_without_precomputed() -> None:
    subset, outcome = make_data()
    with pytest.raises(ValueError):
        assign_folds(subset, outcome, n_folds=N_FOLDS)


def test_assign_folds_cell_smaller_than_folds_is_valid() -> None:
    # (P2, 1) and (P3, 1) have 3 rows against 4 folds: some folds simply
    # lack that cell, which is the NSCH reality for rare outcomes.
    subset, outcome = make_data()
    folds = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    cell = folds[(subset == "P2") & (outcome == 1)]
    assert len(cell) == 3
    assert len(np.unique(cell)) == 3  # dealt to three distinct folds


# --------------------------------------------------------------------------
# iter_soak_splits at full sizes
# --------------------------------------------------------------------------


def collect(splits: Iterable[SoakSplit]) -> list[SoakSplit]:
    out = list(splits)
    assert all(isinstance(s, SoakSplit) for s in out)
    return out


def test_iter_splits_full_count_and_unique_triples() -> None:
    subset, outcome = make_data()
    folds = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    splits = collect(iter_soak_splits(folds, subset, sizes=-1))
    assert len(splits) == 3 * 3 * N_FOLDS  # sources * subsets * folds
    triples = {(s.test_subset, s.fold, s.train_source) for s in splits}
    assert len(triples) == len(splits)
    assert all(not s.downsampled for s in splits)


def test_iter_splits_set_identities_match_definitions() -> None:
    subset, outcome = make_data()
    folds = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    idx = np.arange(len(subset))
    for s in collect(iter_soak_splits(folds, subset, sizes=-1)):
        in_subset = subset == s.test_subset
        in_fold = folds == s.fold
        expected_test = idx[in_subset & in_fold]
        expected_train = {
            TrainSource.SAME: idx[in_subset & ~in_fold],
            TrainSource.OTHER: idx[~in_subset & ~in_fold],
            TrainSource.ALL: idx[~in_fold],
        }[s.train_source]
        assert np.array_equal(s.test_idx, expected_test)
        assert np.array_equal(s.train_idx, expected_train)


def test_iter_splits_train_and_test_disjoint() -> None:
    subset, outcome = make_data()
    folds = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    for s in collect(iter_soak_splits(folds, subset, sizes=-1)):
        assert len(np.intersect1d(s.train_idx, s.test_idx)) == 0


def test_iter_splits_all_is_union_of_same_and_other() -> None:
    subset, outcome = make_data()
    folds = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    splits = collect(iter_soak_splits(folds, subset, sizes=-1))
    by_key = {(s.test_subset, s.fold, s.train_source): s for s in splits}
    for test_subset in np.unique(subset):
        for fold in range(1, N_FOLDS + 1):
            same = by_key[(test_subset, fold, TrainSource.SAME)].train_idx
            other = by_key[(test_subset, fold, TrainSource.OTHER)].train_idx
            union = by_key[(test_subset, fold, TrainSource.ALL)].train_idx
            assert np.array_equal(union, np.union1d(same, other))


def test_iter_splits_order_is_deterministic() -> None:
    subset, outcome = make_data()
    folds = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    a = [(s.test_subset, s.fold, s.train_source) for s in iter_soak_splits(folds, subset)]
    b = [(s.test_subset, s.fold, s.train_source) for s in iter_soak_splits(folds, subset)]
    assert a == b


def test_iter_splits_rejects_mismatched_lengths() -> None:
    subset, outcome = make_data()
    folds = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    with pytest.raises(ValueError):
        collect(iter_soak_splits(folds[:-1], subset))


def test_iter_splits_rejects_unknown_sizes() -> None:
    subset, outcome = make_data()
    folds = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    with pytest.raises(ValueError):
        collect(iter_soak_splits(folds, subset, sizes=2))


# --------------------------------------------------------------------------
# iter_soak_splits with downsampling (sizes 0)
# --------------------------------------------------------------------------


def test_sizes0_keeps_all_full_splits_and_adds_downsampled() -> None:
    subset, outcome = make_data()
    folds = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    full_only = {
        (s.test_subset, s.fold, s.train_source) for s in iter_soak_splits(folds, subset, sizes=-1)
    }
    splits = collect(iter_soak_splits(folds, subset, outcome, sizes=0, seed=SEED))
    full = {(s.test_subset, s.fold, s.train_source) for s in splits if not s.downsampled}
    assert full == full_only
    # Downsampled variants exist exactly for the sources whose nominal
    # size exceeds the per-test-subset minimum: OTHER and ALL here, SAME never.
    nom = nominal_counts(subset, N_FOLDS)
    down = {(s.test_subset, s.train_source) for s in splits if s.downsampled}
    expected = {
        (ts, TrainSource(src))
        for ts, counts in nom.items()
        for src in ("same", "other", "all")
        if counts[src] > min(counts.values())
    }
    assert down == expected


def test_sizes0_downsample_is_subset_of_full_counterpart() -> None:
    subset, outcome = make_data()
    folds = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    splits = collect(iter_soak_splits(folds, subset, outcome, sizes=0, seed=SEED))
    full = {
        (s.test_subset, s.fold, s.train_source): s.train_idx for s in splits if not s.downsampled
    }
    for s in splits:
        if s.downsampled:
            counterpart = full[(s.test_subset, s.fold, s.train_source)]
            assert np.all(np.isin(s.train_idx, counterpart))
            assert np.array_equal(s.test_idx, s.test_idx[np.argsort(s.test_idx)])


def test_sizes0_per_stratum_floor_counts() -> None:
    subset, outcome = make_data()
    folds = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    nom = nominal_counts(subset, N_FOLDS)
    splits = collect(iter_soak_splits(folds, subset, outcome, sizes=0, seed=SEED))
    full = {
        (s.test_subset, s.fold, s.train_source): s.train_idx for s in splits if not s.downsampled
    }
    for s in splits:
        if not s.downsampled:
            continue
        counts = nom[s.test_subset]
        target = min(counts.values())
        own = counts[s.train_source.value]
        counterpart = full[(s.test_subset, s.fold, s.train_source)]
        for label in np.unique(subset):
            for y in (0, 1):
                stratum_full = counterpart[
                    (subset[counterpart] == label) & (outcome[counterpart] == y)
                ]
                kept = s.train_idx[(subset[s.train_idx] == label) & (outcome[s.train_idx] == y)]
                assert len(kept) == len(stratum_full) * target // own


def test_sizes0_reproducible_given_seed() -> None:
    subset, outcome = make_data()
    folds = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    a = collect(iter_soak_splits(folds, subset, outcome, sizes=0, seed=SEED))
    b = collect(iter_soak_splits(folds, subset, outcome, sizes=0, seed=SEED))
    assert len(a) == len(b)
    for sa, sb in zip(a, b, strict=True):
        assert np.array_equal(sa.train_idx, sb.train_idx)


def test_sizes0_changes_with_seed() -> None:
    subset, outcome = make_data()
    folds = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    a = [s for s in iter_soak_splits(folds, subset, outcome, sizes=0, seed=SEED) if s.downsampled]
    b = [
        s for s in iter_soak_splits(folds, subset, outcome, sizes=0, seed=SEED + 1) if s.downsampled
    ]
    assert any(not np.array_equal(sa.train_idx, sb.train_idx) for sa, sb in zip(a, b, strict=True))


def test_sizes0_requires_seed() -> None:
    subset, outcome = make_data()
    folds = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    with pytest.raises(ValueError):
        collect(iter_soak_splits(folds, subset, outcome, sizes=0))


def test_sizes0_requires_outcome() -> None:
    subset, outcome = make_data()
    folds = assign_folds(subset, outcome, n_folds=N_FOLDS, seed=SEED)
    with pytest.raises(ValueError):
        collect(iter_soak_splits(folds, subset, sizes=0, seed=SEED))


# --------------------------------------------------------------------------
# ignore_group_kfold
# --------------------------------------------------------------------------


def test_ignore_group_kfold_test_sets_partition_rows() -> None:
    _, outcome = make_data()
    pairs = list(ignore_group_kfold(outcome, n_folds=N_FOLDS, seed=SEED))
    assert len(pairs) == N_FOLDS
    all_test = np.sort(np.concatenate([test for _, test in pairs]))
    assert np.array_equal(all_test, np.arange(len(outcome)))


def test_ignore_group_kfold_train_is_complement_of_test() -> None:
    _, outcome = make_data()
    idx = np.arange(len(outcome))
    for train, test in ignore_group_kfold(outcome, n_folds=N_FOLDS, seed=SEED):
        assert np.array_equal(np.union1d(train, test), idx)
        assert len(np.intersect1d(train, test)) == 0


def test_ignore_group_kfold_stratifies_outcome() -> None:
    _, outcome = make_data()
    folds = ignore_group_kfold(outcome, n_folds=N_FOLDS, seed=SEED)
    pos_per_fold = [int(outcome[test].sum()) for _, test in folds]
    assert max(pos_per_fold) - min(pos_per_fold) <= 1


def test_ignore_group_kfold_deterministic_given_seed() -> None:
    _, outcome = make_data()
    a = list(ignore_group_kfold(outcome, n_folds=N_FOLDS, seed=SEED))
    b = list(ignore_group_kfold(outcome, n_folds=N_FOLDS, seed=SEED))
    for (ta, sa), (tb, sb) in zip(a, b, strict=True):
        assert np.array_equal(ta, tb)
        assert np.array_equal(sa, sb)


def test_ignore_group_kfold_requires_seed() -> None:
    _, outcome = make_data()
    with pytest.raises(ValueError):
        list(ignore_group_kfold(outcome, n_folds=N_FOLDS))


def test_ignore_group_kfold_rejects_too_few_folds() -> None:
    _, outcome = make_data()
    with pytest.raises(ValueError):
        list(ignore_group_kfold(outcome, n_folds=1, seed=SEED))


# --- Downsample stratification: (subset, outcome), not outcome alone -------
#
# Regression guard for the defect found against the mlr3resampling fixture:
# the sizes=0 downsample strata are the (subset, outcome) pair. For SAME and
# OTHER the train set spans one subset, so the pair collapses to outcome and
# the two rules agree; only ALL can tell them apart. These fixtures are sized
# so the rules give different answers, so the test fails if the strata ever
# revert to outcome alone.
#
# Layout: subset A has 15 No / 5 Yes, subset B has 20 No / 10 Yes, dealt
# round-robin across 5 folds so every fold holds A|No=3, A|Yes=1, B|No=4,
# B|Yes=2. Every ALL train set is therefore A|No=12, A|Yes=4, B|No=16,
# B|Yes=8 (40 rows), with nominal own=40 and target=16.
#
#   (subset, outcome):  12*16//40=4  4*16//40=1  16*16//40=6  8*16//40=3 -> 14
#   outcome alone:      28*16//40=11            12*16//40=4             -> 15

STRATA_FOLDS = 5
STRATA_SEED = 3
STRATA_CELLS = {("A", "No"): 15, ("A", "Yes"): 5, ("B", "No"): 20, ("B", "Yes"): 10}
STRATA_EXPECTED_ALL = {("A", "No"): 4, ("A", "Yes"): 1, ("B", "No"): 6, ("B", "Yes"): 3}


def make_strata_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subset, outcome, and fold arrays with an exactly even round-robin deal."""
    subset: list[str] = []
    outcome: list[str] = []
    folds: list[int] = []
    for (s, y), n in STRATA_CELLS.items():
        subset.extend([s] * n)
        outcome.extend([y] * n)
        folds.extend((np.arange(n) % STRATA_FOLDS + 1).tolist())
    return np.array(subset), np.array(outcome), np.array(folds)


def test_downsample_strata_are_subset_by_outcome() -> None:
    """ALL downsamples floor within each (subset, outcome) cell."""
    subset, outcome, precomputed = make_strata_data()
    fold_ids = assign_folds(
        subset=subset,
        outcome=outcome,
        n_folds=STRATA_FOLDS,
        precomputed=precomputed,
    )
    seen = 0
    for split in iter_soak_splits(
        fold_ids=fold_ids,
        subset=subset,
        outcome=outcome,
        sizes=0,
        seed=STRATA_SEED,
    ):
        if not (split.downsampled and split.train_source is TrainSource.ALL):
            continue
        seen += 1
        got = {
            (s, y): int(np.sum((subset[split.train_idx] == s) & (outcome[split.train_idx] == y)))
            for (s, y) in STRATA_CELLS
        }
        assert got == STRATA_EXPECTED_ALL
        # 14 under (subset, outcome); 15 if the strata regress to outcome alone.
        assert len(split.train_idx) == 14
    # Both subsets, every fold.
    assert seen == 2 * STRATA_FOLDS


def test_downsample_unchanged_for_single_subset_sources() -> None:
    """SAME and OTHER span one subset, so the finer strata are inert."""
    subset, outcome, precomputed = make_strata_data()
    fold_ids = assign_folds(
        subset=subset,
        outcome=outcome,
        n_folds=STRATA_FOLDS,
        precomputed=precomputed,
    )
    seen = 0
    for split in iter_soak_splits(
        fold_ids=fold_ids,
        subset=subset,
        outcome=outcome,
        sizes=0,
        seed=STRATA_SEED,
    ):
        if not split.downsampled or split.train_source is TrainSource.ALL:
            continue
        seen += 1
        # Only subset B's rows are in play: B SAME (test B) and A OTHER
        # (test A) are both B|No=16, B|Yes=8, own=24, target=16.
        assert set(subset[split.train_idx]) == {"B"}
        counts = {y: int(np.sum(outcome[split.train_idx] == y)) for y in ("No", "Yes")}
        assert counts == {"No": 10, "Yes": 5}
        assert len(split.train_idx) == 15
    assert seen == 2 * STRATA_FOLDS
