# Changelog

All notable changes to this project are documented here.

This mirrors the R `nsch` package's `NEWS.md`: one `## YYYY.M.DD (PR#NN)`
section per PR, newest first. Versions follow a date-based scheme, `YYYY.M.DD`,
bumped per PR to the date it lands. When two PRs land on the same day, the
second and later append a micro segment (`YYYY.M.DD.MICRO`, e.g. `2026.6.29.1`)
so each version stays unique and the date stays honest.

## 2026.8.6 (PR#NN)

- Added `analyses/soak_ttests.py`: the two-sided paired t on 9 degrees of freedom from Hocking et al., applied per test subset to both the R reference and our results, with the two summary figures. Mean AUC on the 2020 test subset comes out at 0.9679 training on All and 0.9664 on Same, against 0.9670 and 0.9658 in the published paper.
- Added `analyses/soak_criteria.py`: compares three candidate replication criteria and recomputes verdict agreement under Bonferroni and Benjamini-Hochberg.
- Added `docs/replication-equivalence.md`: what is identical between the two implementations, what differs and why, and the case for judging replication on contrast estimates rather than significance verdicts.
- Ruled out two hypotheses for the verdict disagreements. Penalty family is not the cause (ridge, lasso and a 60-point ridge grid all land within 0.0006 of each other). Grid coarseness is not the cause either.
- Added `analyses/r_vs_r.py`: clusters the ten available R runs of the same analysis by pairwise AUC distance. The `mlr3learners` version moves results by 0.0073 mean absolute AUC, against 0.0005 for the inner seed or the machine. Leaving the `cv.glmnet` seed unset is bit-identical to setting it to 1. Against R runs on the current build the Python port sits at 0.0021, four times the within-build floor. Two of four SOAK verdicts differ between R runs, so verdict agreement is not a standard the reference meets against itself.

## 2026.8.5 (PR#12)

- Added `analyses/run_glmnet_replication.py` and `analyses/probe_glmnet_split.py`: penalized logistic regression fitted on all 60 non-downsampled SOAK splits, compared against `classif.cv_glmnet` in `reproduce-soak-nsch/results/2026-03-06/NSCH_proj.csv`. All 60 splits match the R reference. The mean paired difference in test AUC is +0.00017 across the 20 test folds (sd 0.00865, sem 0.00193, t = 0.09), so there is no detectable systematic difference in discrimination.
- The comparison is paired: R and Python evaluate identical test rows, verified index-for-index in PR#9. Differences cluster by test fold rather than by split, since the three train sources within a `(subset, fold)` share a test set, so the summary aggregates to the fold. Test folds carry roughly 55 positive cases, which accounts for the per-fold scatter.
- Committed `analyses/glmnet_replication.csv` as the result artifact so the numbers can be checked without a three-minute rerun.

## 2026.8.4 (PR#9)

- Fixed `iter_soak_splits`: the `sizes=0` downsample now stratifies on `(subset, outcome)` rather than on outcome alone. Verified against the archived `ResamplingSameOtherSizesCV` instance for NSCH_autism (46,010 rows, folds=10, sizes=0, seed=1): all 100 iteration keys, all 60 full splits index-for-index, and all 40 downsampled train sizes and per-stratum counts now agree with R. The strata differ only for `ALL`, whose train set spans more than one subset; `SAME` and `OTHER` are unaffected. Python's own downsampled row selection for `ALL` changes as a result, since four strata are drawn instead of two.
- Added `analyses/`: `make_split_digest.py` reports three-tier agreement with the R fixture, and `diagnose_downsample_strata.py` recovers the downsample rule from the fixture alone.

## 2026.7.8 (PR#4)

- Added `nsch_ml.soak`: stratified fold assignment with a precomputed passthrough for reproducing the R analysis's folds, the same/other/all split iterator with the `sizes=0` stratified downsampling, and the inner ignore-group k-fold. Semantics pinned against the mlr3resampling source (archived `ResamplingSameOtherCV` 2024.9.6; current `ResamplingSameOtherSizesCV`).
- Changed the pre-commit mypy hook to run the project's mypy through uv, so pre-commit, CI, and the local gate share one mypy version, environment, and config.

## 2026.7.7 (PR#1)

- Initial repository scaffolding:
  - `pyproject.toml` with `[project]` metadata, the analysis dependency stack, and configuration for ruff, mypy, pytest, and coverage.
  - GitHub Actions CI workflow with lint, matrix test (Python 3.11–3.13 on Linux; 3.13 on macOS and Windows; a lowest-direct floor check), build, docs, and dependency-review jobs.
  - GitHub Actions docs workflow publishing the mkdocs site to GitHub Pages.
  - Pre-commit hooks mirroring CI checks, Dependabot configuration, issue and PR templates, and `CODEOWNERS`.
  - Documentation skeleton (`mkdocs.yml`, `docs/index.md`, `docs/design-decisions.md`).
  - Planning documents under `planning/`: the development plan and the Workstream C scoping document.
  - The `src/nsch_ml/` package with a smoke test.
