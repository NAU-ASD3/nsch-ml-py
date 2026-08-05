"""SOAK resampling: fold assignment, same/other/all splits, and the inner CV.

The mechanics replicate mlr3resampling's ResamplingSameOtherCV (fold
assignment and split logic; archived at version 2024.9.6, removed from
the package by 2026.5.19) and ResamplingSameOtherSizesCV (the sizes=0
downsampling used by the fairness analysis). See the function docstrings
and docs/design-decisions.md for what is replicated exactly versus
statistically.
"""

from nsch_ml.soak.splitter import (
    SoakSplit,
    TrainSource,
    assign_folds,
    ignore_group_kfold,
    iter_soak_splits,
)

__all__ = [
    "SoakSplit",
    "TrainSource",
    "assign_folds",
    "ignore_group_kfold",
    "iter_soak_splits",
]
