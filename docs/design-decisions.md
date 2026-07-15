# Design decisions

Why the package is built the way it is. Each entry records a decision a maintainer might otherwise re-litigate. Longer treatments live in `planning/workstream-c-scoping.md`; these are the short versions.

## Equivalence is judged on predictions, not coefficients

glmnet and scikit-learn standardize inputs differently and parameterize penalty strength differently, so coefficient-level agreement between the R analysis and this port is unreachable in general. The replication contract is therefore held-out predicted probabilities within an agreed tolerance, plus agreement on the derived quantities (AUC, accuracy, and the same/other/all error-difference test). Scoping doc §4 has the full argument.

## We wrote our own SOAK splitter

soakpy (PyPI 0.0.54) stratifies on the subset only rather than the (subset, outcome) pair, its downsampling isn't seeded, and its high-level API is regression-only. Our splitter stratifies on (subset, outcome), seeds the downsample so the fairness runs are reproducible at both size settings, and returns index arrays that plug into scikit-learn and XGBoost directly. soakpy remains a cross-check on the plain partitioning, nothing more.

## The linear learner is penalized logistic regression, not Ridge

The original analysis fits binomial `cv_glmnet` models. The faithful scikit-learn analogue is `LogisticRegression(penalty='l2')` (ridge) or `penalty='l1'` (lasso, used by the fairness driver). `Ridge` and `RidgeClassifier` minimize squared error on the 0/1 labels, a different objective that would quietly break the prediction-level comparison. The `C`-to-lambda mapping and the standardization handling are documented alongside the learner wrappers.

## svy covers descriptive statistics only, for now

The replication's cross-validation is unweighted by design, matching the original analysis, so `svy` is needed only for design-based descriptive numbers. The survey-weighted extension will lean on it much harder; its installability is an open question tracked in the planning docs, and it stays out of `pyproject.toml` until that's settled.

## Fixtures are fetched, not committed

The NSCH_autism splitter-validation fixture is 46,010 rows by 364 columns, too heavy for the repo. Tests that need it download it from the Zenodo record (10.5281/zenodo.18273949) with a pinned checksum, cache it under `fixtures/cache/` (gitignored), and skip when offline (`network` marker). The one-line golden header from the R model matrix is tiny and is committed.
