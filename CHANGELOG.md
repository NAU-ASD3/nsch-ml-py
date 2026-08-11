# Changelog

All notable changes to this project are documented here.

This mirrors the R `nsch` package's `NEWS.md`: one `## YYYY.M.DD (PR#NN)`
section per PR, newest first. Versions follow a date-based scheme, `YYYY.M.DD`,
bumped per PR to the date it lands. When two PRs land on the same day, the
second and later append a micro segment (`YYYY.M.DD.MICRO`, e.g. `2026.6.29.1`)
so each version stays unique and the date stays honest.

## 2026.8.10.3 (PR#20)

- Planning notes are no longer tracked. `planning/` holds working documents that name people, describe one machine's filesystem, and go stale faster than anything in `docs/`. They stay on disk and are gitignored.
- Refreshed the entry-point documents. `README.md` now says why the port exists rather than only what it contains, and `docs/index.md` names the three published documents and which to read first. Both dropped links into `planning/` that would have 404'd.
- Removed SHAP from the package description in `README.md`, `docs/index.md`, and the package docstring. It remains in `pyproject.toml` as a dependency and a keyword, pending a decision on whether it is cancelled or deferred.

## 2026.8.10.2 (PR#19)

- Added `analyses/run_featureless_replication.py`, a port of the baseline learner that ignores every feature and predicts the training set's outcome rate. Accuracy matches the published run exactly in all six cells, difference 0.000000.
- The featureless comparison tests everything except the model. Its predictions depend only on which children are in the training set and how many have the outcome, so exact agreement confirms the data, the outcome coding, and the split membership against the publication. It is also unaffected by the fold-assignment differences between runs, since accuracy at this base rate is a property of the data rather than the partition.
- AUC is 0.5 on all 60 splits, as it must be for a constant prediction, and is reported only to confirm both sides agree the ordering is degenerate.

## 2026.8.10.1 (PR#18)

- `analyses/soak_ttests.py` now reports percent error alongside AUC. The SOAK paper's own figures compute their p-values on percent error, and we had been applying the method to AUC. Both come from accuracy columns already in the results file, so no rerun was needed.
- On this dataset percent error cannot see what AUC sees. The strongest contrast in the R results, 2019 All-minus-Same, is p = 0.0015 on AUC and p = 0.3662 on error. The implementation gap is 3.5 times the size of the contrasts on error against 0.5 times on AUC. At a 3% base rate, accuracy runs from 0.973 to 0.980 across every split. AUC stays primary; percent error is reported because it is the paper's metric.
- Every figure in `docs/replication-equivalence.md` now comes from one run, the lasso fits in `glmnet_replication_lasso_seed1_is100.csv`. An earlier draft mixed a ridge run and a lasso run in the same tables.
- The AUC offset is characterised rather than left open: a level shift of about +0.001 across all six cells, varying by ±0.0003, which cancels in the contrasts because a contrast is a difference between cells.
- Rewrote `docs/replication-equivalence.md` for readers without background on the prior work, adding sections on the study, the SOAK design, and the model.
- Removed two claims that could not be traced to output: a `lambda.min` Spearman figure and a three-way penalty-family comparison.

## 2026.8.10 (PR#17)

- Corrected the central claim of `docs/replication-equivalence.md`. Earlier drafts said the `mlr3learners` build shifts the R results by 0.0073 mean absolute AUC, larger than the effects the study reports. The R runs it came from do not all partition the data the same way, so pairing them fold by fold compares different children. On cell means, which do not depend on fold numbering, all eleven runs agree to within 0.0005.
- Added `analyses/diagnose_fold_relabelling.py`. Optimal matching closes 64 to 66 percent of the between-group per-fold gap and 5 to 23 percent within a group, with no overlap across 44 pairs. Batchtools has a third fold assignment, which explains an anomaly the docs had listed as open.
- Added `analyses/cell_mean_distances.py`, which compares runs on a quantity that does not depend on fold numbering and warns that its distances are not on the same scale as the per-fold ones.
- Added `analyses/compare_published_registry.py`. The results the SOAK paper was published from are in the cv-same-other-paper checkout and reproduce Section 4.3 to four decimals. They used `same_other_cv`, so only cell means are comparable, and on those they sit in the middle of the pack.
- Recorded that our fold fixture was drawn by `mlr3resampling` 2026.5.19 under seed 1, and that our folds match `NSCH_seed1` at row level rather than by assumption.
- `docs/equivalence-margin.md` gained a dated addendum rather than an edit. Two statements in it relied on the 0.0073 figure; the margins themselves were anchored on runs sharing a fold assignment and are unaffected.
- Recorded Olivia's decision that care coordination is a predictor rather than an outcome, which closes the last open question on the target side.

## 2026.8.8 (PR#16)

- Ran the prediction-level comparison against the margins committed in #14. Fifty-eight of sixty splits clear every gated check. Two do not, and they are one fitted model scored on its two test subsets.
- Added `analyses/prediction_equivalence.py`, which gates on Spearman of held-out scores, probability MAD, and fold-level AUC, and refuses to compare unless both files cover the same rows and agree on the outcome for every one.
- Added `analyses/diagnose_failing_split.py`. The failing model holds out an ordinary fold with an ordinary feature count; the children ranked most differently are all true negatives below 0.007, several of them tied in R and untied in Python. Spearman over the full held-out set spends its power on the 97% the model is confident about, which makes it a poor gate for a 3% outcome. The margin is not revised.
- Switched the lasso path to `liblinear`, which fits the same L1 objective by coordinate descent, the family glmnet uses, in about a fifth of `saga`'s time. `--l1-solver` selects between them.
- Added `--intercept-scaling`. `liblinear` penalizes the intercept where glmnet and `saga` do not; `analyses/probe_intercept_scaling.py` measures the cost, and setting it to 100 tightened fold-level agreement by roughly a quarter. Both runs are kept as evidence.
- Added `--save-predictions`, so a two-hour run yields the per-row probabilities the comparison needs rather than discarding them.
- Moved to scikit-learn's `l1_ratio` spelling, deprecating `penalty`, which had been emitting an inconsistency warning that the blanket FutureWarning filter was hiding. `ConvergenceWarning` is no longer suppressed either.

## 2026.8.7.1 (PR#15)

- Recovered R's predicted probabilities and the intercept, which the coefficient files could not supply. `proj_grid` takes a `save_pred` argument defaulting to `FALSE`, and the package's default glmnet saver drops the intercept with `coef(x$model)[-1, ]`; both are one argument each.
- Added `analyses/repair_r_predictions.py`, which corrects two class-label conventions in R's export and verifies the result against both R's reported AUC and the outcome prevalence.
- Added `analyses/inspect_r_coefficients.py`, reporting sparsity, the features carrying most weight, and whether the household identifier or survey weight survive the penalty.
- Corrected the record on penalty family. `docs/design-decisions.md` said the core analysis used ridge with lasso only in the fairness driver. `cv.glmnet` defaults to `alpha = 1`, and R's saved coefficients confirm lasso, so the primary comparison has been ridge against lasso.
- Documented three positive-class conventions, none visible in an AUC, each of which would independently invert a probability-scale comparison: the design's `y` holds "Yes"/"No" rather than 1/0, mlr3 assigned "No" as positive, and scikit-learn's `roc_auc_score` takes the lexicographically last string label as positive.
- Revised `docs/design-decisions.md` and `CONTRIBUTING.md`, including a note on narrowing Polars reducer types under mypy and a section on conventions for the `analyses/` scripts.

## 2026.8.7 (PR#14)

- Recovered R's fitted models from the saved learner objects. `LearnerClassifCVGlmnetSave` stores the 364 coefficients at the selected lambda, and `analyses/verify_r_coefficients.py` confirms they reproduce R's reported AUC on all 60 splits to 1e-14. Coefficient and ranking comparisons therefore need no further R runs; probability-scale work still needs the intercept, which is not stored.
- glmnet's coefficients are oriented toward `y = 0` in this fixture. Every split reproduces exactly `1 - AUC` before the sign is corrected, so any coefficient comparison must account for it or report 364 wrong signs per split.
- R's `cv_glmnet` runs lasso, 65 nonzero of 364. `Unique_Household_ID` and `Selected_Child_Weight` are dropped in all 60 splits, so the published analysis neither leaks the household identifier nor uses the survey weight as a predictor.
- For any fold, "same" on 2019 and "other" on 2020 train on identical rows and their fitted coefficients are bit-identical, so the 60 splits hold 30 distinct models.
- Added `analyses/r_seed_yardstick.py`. Three R runs differing only in the inner CV seed disagree by 0.004 mean absolute on coefficients, bottom out at 0.965 Spearman on held-out rankings, and agree to 0.0006 on fold-level AUC. The finer the quantity, the less reproducible the analysis is against its own seed.
- Added `docs/equivalence-margin.md`, fixing the pass/fail standard before the comparison script was written. Committed in `313489c` directly to `main` under admin bypass so the timestamp would precede any comparison; the bypass is recorded in GitHub's audit log and noted here rather than left to be discovered.

## 2026.8.6 (PR#13)

- Added `analyses/soak_ttests.py`: the two-sided paired t on 9 degrees of freedom from Hocking et al., applied per test subset to both the R reference and our results, with two summary figures. Mean AUC on the 2020 test subset agrees with the published figures to about a thousandth.
- Added `analyses/soak_criteria.py`: compares three candidate replication criteria and recomputes verdict agreement under Bonferroni and Benjamini-Hochberg. Contrast estimates agree in every comparison; significance verdicts do not, and which one disagrees depends on the multiplicity adjustment.
- Added `analyses/r_vs_r.py`: clusters the ten available R runs of the same analysis by pairwise AUC distance. The `mlr3learners` build moves results by an order of magnitude more than the inner seed or the machine does, and two of four SOAK verdicts differ between R runs.
- Added `docs/replication-equivalence.md`, which is the single source for the equivalence numbers. Scripts point at it rather than restating figures that go stale.
- Widened the penalty grid in `run_glmnet_replication.py` from 12 points to 60. Together with the lasso run kept in `glmnet_replication_lasso.csv`, this rules out penalty family and grid coarseness as causes of the verdict disagreements.
- Fixed the implementation half-width in `soak_criteria.py`, which used the upper interval bound rather than `(hi - lo) / 2` and so overstated the ratio against the contrast size.
- Fixed stale defaults in `soak_ttests.py`, which pointed at the pre-grid-widening results file and the wrong AUC column, so running it as documented reproduced numbers the doc contradicted.
- Both scripts now warn when a results file has fewer folds than expected, instead of silently comparing a partial run against a complete one.
- Renamed throughout `analyses/` for descriptive names; single letters are now reserved for range indices and the `df`/`ax` conventions. The convention is recorded in `CONTRIBUTING.md`. Reusing short names for two types is what produced the type collisions this pass fixed.
- Added `tests/test_analyses.py`: 23 tests covering the Benjamini-Hochberg adjustment and the run clustering, both of which feed stated results.
- `pyproject.toml`: removed lint and mypy configuration referencing rule families the repo does not select, extended mypy to `analyses` and `tests`, added overrides for scipy and matplotlib.
- `.pre-commit-config.yaml`: ruff and mypy now run through `uv run` rather than a pinned hook binary and a hardcoded path. The pinned ruff had drifted ten minor versions behind the dev dependency; the mypy hook's `src/` argument overrode `files` in `pyproject.toml`, so it had never checked `analyses/` or `tests/`.
- Reran the comparison against `NSCH_seed1`, a reference from the installed `mlr3learners` build, and made it primary; `analyses/glmnet_replication_seed1.csv` is committed alongside the earlier `grid60` file, which used a February reference. The measured implementation gap falls from about four times the size of the SOAK contrasts to roughly one to one.
- The criterion question is deferred rather than answered. Three candidates were evaluated on fold-level AUC and each fails differently; the doc now records that evidence and names the plan of record, a prediction-level comparison against a margin fixed before it is computed. One cell (2020 Other) shows a raw p of 0.012 that does not survive multiplicity correction.

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
