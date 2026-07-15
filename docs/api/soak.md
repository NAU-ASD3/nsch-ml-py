# soak

The SOAK splitter: stratified fold assignment, the same/other/all split
iterator, the seeded downsampling used by the fairness analysis, and the
inner ignore-group k-fold. What is replicated exactly versus statistically
relative to the R implementation is covered in
[Design decisions](../design-decisions.md).

::: nsch_ml.soak
    options:
      members:
        - TrainSource
        - SoakSplit
        - assign_folds
        - iter_soak_splits
        - ignore_group_kfold
