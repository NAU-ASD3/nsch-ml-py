"""The SOAK splitter: stratified fold assignment and same/other/all splits.

Given a subset label (the survey period, or a fairness grouping) and a
binary outcome for every row, this module produces the train/test index
arrays for Same/Other/All K-fold cross-validation, plus the seeded,
stratified downsampling that the fairness analysis runs at ``sizes=0``,
and the plain stratified k-fold used inside hyperparameter searches.

Equivalence with the R implementation is split in two, deliberately.
The mapping from a fold assignment to same/other/all index sets is pure
set logic and must match mlr3resampling exactly (validated against the
R fold assignments in the equivalence tests). The random parts, fold
assignment and downsampling, cannot match R draw-for-draw because R's
``sample()`` and NumPy's generator are different RNGs; they replicate
the R rules (stratification, per-stratum floor counts) and are seeded
for our own reproducibility, with R's actual assignments importable via
``precomputed`` when the study's exact folds are needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.model_selection import StratifiedKFold

if TYPE_CHECKING:
    from collections.abc import Iterator

IntArray = npt.NDArray[np.int64]
"""Sorted int64 row indices, ready for sklearn/xgboost fancy indexing."""


class TrainSource(Enum):
    """Which rows a SOAK split trains on, relative to the test subset."""

    SAME = "same"
    OTHER = "other"
    ALL = "all"


@dataclass(frozen=True)
class SoakSplit:
    """One SOAK train/test split.

    Attributes
    ----------
    test_subset
        Label of the subset being tested (subset labels are coerced to
        ``str`` on entry, so period ``P4`` and fixture year ``2019``
        both come back as strings).
    fold
        The held-out fold, ``1..n_folds``.
    train_source
        SAME, OTHER, or ALL, per the SOAK definitions.
    downsampled
        True for the ``sizes=0`` variants whose train set was reduced
        toward the per-test-subset minimum of the nominal train sizes.
        The full counterpart of every downsampled split is also
        yielded.
    train_idx, test_idx
        Sorted int64 row indices into the original arrays.
    """

    test_subset: str
    fold: int
    train_source: TrainSource
    downsampled: bool
    train_idx: IntArray
    test_idx: IntArray


def _as_array(x: npt.NDArray[Any] | pl.Series) -> npt.NDArray[Any]:
    """Polars Series to NumPy at the door; everything else through asarray."""
    if isinstance(x, pl.Series):
        return x.to_numpy()
    return np.asarray(x)


def _require_same_length(*arrays: npt.NDArray[Any]) -> int:
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        msg = f"inputs must be the same length, got lengths {sorted(lengths)}"
        raise ValueError(msg)
    return lengths.pop()


def _as_fold_ints(values: npt.NDArray[Any], n_folds: int) -> IntArray:
    """Validate fold values: integer-valued and within 1..n_folds."""
    arr = np.asarray(values)
    if not np.issubdtype(arr.dtype, np.integer) and not np.all(arr == np.floor(arr)):
        msg = "fold values must be integers"
        raise ValueError(msg)
    out = arr.astype(np.int64)
    if out.min() < 1 or out.max() > n_folds:
        msg = f"fold values must lie in 1..{n_folds}, got range [{out.min()}, {out.max()}]"
        raise ValueError(msg)
    return out


def assign_folds(
    subset: npt.NDArray[Any] | pl.Series,
    outcome: npt.NDArray[Any] | pl.Series,
    *,
    n_folds: int = 10,
    seed: int | None = None,
    precomputed: npt.NDArray[Any] | pl.Series | None = None,
) -> IntArray:
    """Assign each row a fold in ``1..n_folds``, stratified on (subset, outcome).

    Every (subset, outcome) cell is shuffled and dealt across the folds
    round-robin, so per-cell fold counts differ by at most one. This is
    the rule ResamplingSameOtherCV applied via its per-stratum sampler;
    the draw itself will not match R's for any seed (different RNGs),
    which is why ``precomputed`` exists: pass the fold column exported
    from the R analysis to reproduce the study's exact folds, mirroring
    mlr3resampling's own support for a user-supplied fold role.

    Parameters
    ----------
    subset
        Subset label per row (period, or a fairness grouping).
    outcome
        Binary outcome per row.
    n_folds
        Number of folds. Must be at least 2.
    seed
        Seed for the per-cell shuffles. Required unless ``precomputed``
        is given.
    precomputed
        An existing fold assignment to validate and pass through.
        Values must be integers in ``1..n_folds``.

    Returns
    -------
    IntArray
        Fold id per row, aligned with the inputs.

    Raises
    ------
    ValueError
        If lengths disagree, ``n_folds < 2``, ``seed`` is missing when
        needed, or ``precomputed`` contains values outside
        ``1..n_folds`` or non-integers.
    """
    subset_arr = _as_array(subset).astype(str)
    outcome_arr = _as_array(outcome)
    n = _require_same_length(subset_arr, outcome_arr)
    if n_folds < 2:
        msg = f"n_folds must be at least 2, got {n_folds}"
        raise ValueError(msg)

    if precomputed is not None:
        given = _as_array(precomputed)
        if len(given) != n:
            msg = f"precomputed has length {len(given)}, expected {n}"
            raise ValueError(msg)
        return _as_fold_ints(given, n_folds)

    if seed is None:
        msg = "seed is required when precomputed is not given"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    folds = np.empty(n, dtype=np.int64)
    # Deterministic cell order (sorted labels), shuffled deal within each
    # cell: counts per fold differ by at most one inside every cell.
    for s in np.unique(subset_arr):
        for y in np.unique(outcome_arr):
            cell = np.flatnonzero((subset_arr == s) & (outcome_arr == y))
            dealt = np.arange(len(cell)) % n_folds + 1
            folds[cell[rng.permutation(len(cell))]] = dealt
    return folds


def iter_soak_splits(
    fold_ids: npt.NDArray[Any] | pl.Series,
    subset: npt.NDArray[Any] | pl.Series,
    outcome: npt.NDArray[Any] | pl.Series | None = None,
    *,
    sizes: int = -1,
    seed: int | None = None,
) -> Iterator[SoakSplit]:
    """Yield every SOAK split implied by a fold assignment.

    For each test subset ``s`` and fold ``k``: test rows are fold ``k``
    within ``s``; SAME trains on the other folds within ``s``; OTHER on
    the other folds outside ``s``; ALL on the other folds everywhere.
    This mapping is deterministic given ``fold_ids`` and must match the
    R implementation index-for-index.

    ``sizes`` follows ResamplingSameOtherSizesCV. At ``-1``, only the
    full train sets are yielded: ``3 * n_subsets * n_folds`` splits. At
    ``0``, each source is additionally yielded downsampled toward the
    per-test-subset minimum of the nominal train sizes, where the
    nominal sizes are computed once per test subset as in the R source:
    ``same = floor(full * (K-1)/K)``, ``all = sum(same)``,
    ``other = all - same``. A source whose nominal size already equals
    that minimum (typically SAME) is not duplicated. The downsample
    replicates the R rule: within each ``(subset, outcome)`` stratum of
    the actual train set, rows are shuffled and a proportional prefix
    kept, with per-stratum counts of
    ``floor(L_s * target / nominal_own)``, so the result preserves both
    subset and outcome balance and its total may fall slightly short of
    the target. Stratifying on the pair rather than on outcome alone
    changes nothing for SAME and OTHER, whose train sets span a single
    subset, but is what makes ALL's counts match mlr3resampling
    exactly. Each downsampled split draws from an
    independent stream derived from ``seed`` and the split's identity,
    so results do not depend on iteration order; the draw is
    reproducible here but does not reproduce R's row selection.

    Parameters
    ----------
    fold_ids
        Fold assignment per row, from :func:`assign_folds`.
    subset
        Subset label per row, aligned with ``fold_ids``.
    outcome
        Binary outcome per row. Required when ``sizes == 0``, where it
        defines the strata the downsample preserves; unused otherwise.
    sizes
        ``-1`` for full train sets only; ``0`` to add the downsampled
        variants.
    seed
        Downsampling seed. Required when ``sizes == 0``.

    Yields
    ------
    SoakSplit
        One split per (test subset, fold, source, size variant).

    Raises
    ------
    ValueError
        If lengths disagree, ``sizes`` is not ``-1`` or ``0``, or
        ``sizes == 0`` without ``seed`` and ``outcome``.
    """
    folds_arr = _as_array(fold_ids).astype(np.int64)
    subset_arr = _as_array(subset).astype(str)
    _require_same_length(folds_arr, subset_arr)
    if sizes not in (-1, 0):
        msg = f"sizes must be -1 (full) or 0 (add downsampled), got {sizes}"
        raise ValueError(msg)
    outcome_arr: npt.NDArray[Any] | None = None
    down_seed = 0
    if sizes == 0:
        if seed is None:
            msg = "seed is required when sizes == 0"
            raise ValueError(msg)
        if outcome is None:
            msg = "outcome is required when sizes == 0"
            raise ValueError(msg)
        down_seed = seed
        outcome_arr = _as_array(outcome)
        _require_same_length(folds_arr, outcome_arr)

    labels = np.unique(subset_arr)
    outcome_levels = np.unique(outcome_arr) if outcome_arr is not None else np.asarray([])
    n_folds = int(folds_arr.max())

    # Nominal train sizes per test subset, transcribed from the R source
    # (ResamplingSameOtherSizesCV): integer truncation included.
    full_count = {s: int(np.sum(subset_arr == s)) for s in labels}
    same_nom = {s: full_count[s] * (n_folds - 1) // n_folds for s in labels}
    all_nom = sum(same_nom.values())
    nominal = {
        s: {
            TrainSource.SAME: same_nom[s],
            TrainSource.OTHER: all_nom - same_nom[s],
            TrainSource.ALL: all_nom,
        }
        for s in labels
    }

    for s_i, s in enumerate(labels):
        in_subset = subset_arr == s
        target = min(nominal[s].values())
        for fold in range(1, n_folds + 1):
            in_fold = folds_arr == fold
            test_idx = np.flatnonzero(in_subset & in_fold).astype(np.int64)
            train_masks = {
                TrainSource.SAME: in_subset & ~in_fold,
                TrainSource.OTHER: ~in_subset & ~in_fold,
                TrainSource.ALL: ~in_fold,
            }
            for src_i, source in enumerate(TrainSource):
                train_idx = np.flatnonzero(train_masks[source]).astype(np.int64)
                yield SoakSplit(
                    test_subset=str(s),
                    fold=fold,
                    train_source=source,
                    downsampled=False,
                    train_idx=train_idx,
                    test_idx=test_idx,
                )
                own = nominal[s][source]
                if sizes == 0 and outcome_arr is not None and own > target:
                    # Independent stream per split identity: reproducible,
                    # order-independent, seed-sensitive.
                    rng = np.random.default_rng([down_seed, s_i, fold, src_i])
                    kept: list[npt.NDArray[np.int64]] = []
                    train_subset = subset_arr[train_idx]
                    train_outcome = outcome_arr[train_idx]
                    for cell_subset in labels:
                        for y in outcome_levels:
                            in_cell = (train_subset == cell_subset) & (train_outcome == y)
                            stratum = train_idx[in_cell]
                            if len(stratum) == 0:
                                continue
                            keep_n = len(stratum) * target // own
                            perm = rng.permutation(len(stratum))
                            kept.append(stratum[perm[:keep_n]])
                    down_idx = np.sort(np.concatenate(kept)).astype(np.int64)
                    yield SoakSplit(
                        test_subset=str(s),
                        fold=fold,
                        train_source=source,
                        downsampled=True,
                        train_idx=down_idx,
                        test_idx=test_idx,
                    )


def ignore_group_kfold(
    outcome: npt.NDArray[Any] | pl.Series,
    *,
    n_folds: int = 5,
    seed: int | None = None,
) -> Iterator[tuple[IntArray, IntArray]]:
    """Plain stratified k-fold that ignores subset and group structure.

    The inner tuning CV, mirroring ResamplingIgnoreGroupCV: used inside
    the XGBoost and kNN hyperparameter searches, where the SOAK subset
    structure is deliberately not respected. A thin wrapper over
    scikit-learn's StratifiedKFold so callers never touch sklearn's API
    directly and seed handling stays uniform across the package.

    Parameters
    ----------
    outcome
        Binary outcome per row; the stratification target.
    n_folds
        Number of folds. Must be at least 2.
    seed
        Shuffle seed. Required.

    Yields
    ------
    tuple[IntArray, IntArray]
        Sorted ``(train_idx, test_idx)`` per fold; the test sets
        partition all rows.

    Raises
    ------
    ValueError
        If ``n_folds < 2`` or ``seed`` is missing.
    """
    if n_folds < 2:
        msg = f"n_folds must be at least 2, got {n_folds}"
        raise ValueError(msg)
    if seed is None:
        msg = "seed is required"
        raise ValueError(msg)
    outcome_arr = _as_array(outcome)
    placeholder = np.zeros((len(outcome_arr), 1))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train, test in skf.split(placeholder, outcome_arr):
        yield (
            np.sort(train).astype(np.int64),
            np.sort(test).astype(np.int64),
        )
