# Design decisions

Why the package is built the way it is. Each entry records a decision a maintainer might otherwise re-litigate. Longer treatments live in `planning/workstream-c-scoping.md` and in the documents named below; these are the short versions.

## Equivalence is judged on predictions, not coefficients

glmnet and scikit-learn standardize inputs differently and parameterize penalty strength differently, so coefficient-level agreement between the R analysis and this port was never going to be reachable. That was the a priori argument. There is now a measurement behind it.

Three R runs differing only in the `cv.glmnet` inner seed, same machine, same afternoon, same package build, disagree with each other by 0.004 mean absolute on the fitted weights, by as much as 0.377 on a single weight where the largest weights run to about 1.04, and on which features are selected at all for roughly two percent of the 364. The reference does not have coefficient stability, so requiring it of a reimplementation would be measuring the lasso's behaviour and calling it our error.

The replication contract is held-out predicted probabilities and score rankings against margins fixed in advance. Those margins, the R-internal yardstick they are anchored to, and what is gated versus merely reported are in `docs/equivalence-margin.md`. What differs between the two implementations and why is in `docs/replication-equivalence.md`.

## We wrote our own SOAK splitter

soakpy (PyPI 0.0.54) stratifies on the subset only rather than the (subset, outcome) pair, its downsampling isn't seeded, and its high-level API is regression-only. Our splitter stratifies on (subset, outcome), seeds the downsample so the fairness runs are reproducible at both size settings, and returns index arrays that plug into scikit-learn and XGBoost directly. soakpy remains a cross-check on the plain partitioning, nothing more.

## The linear learner is penalized logistic regression, and the penalty is lasso

The original analysis fits binomial `cv_glmnet` models. The faithful scikit-learn analogue is `LogisticRegression` with an explicit penalty, not `Ridge` or `RidgeClassifier`, which minimize squared error on the 0/1 labels and would quietly break a prediction-level comparison.

An earlier version of this entry said the core analysis used ridge and that lasso appeared only in the fairness driver. That was wrong. `cv.glmnet` defaults to `alpha = 1`, and R's saved coefficients confirm it: 65 of 364 weights are nonzero on a representative split, 255 features are never selected in any of the 60 splits, and 15 are selected in all of them. The core analysis is lasso, and a ridge port compared against it tests a choice we made rather than the fidelity of the port.

The `C`-to-lambda mapping and the standardization handling are documented alongside the learner wrappers. Two things are worth knowing before running anything. scikit-learn's `saga` solver took eleven and a half hours on the 60 splits where `liblinear`, which is coordinate descent like glmnet's, took 72 minutes; the two agree on AUC to 0.0006 and select penalties within a step or two of each other. And `liblinear` penalizes the intercept where glmnet and `saga` do not, so `intercept_scaling` is set to 100, which brings the intercept to within 0.002 of `saga`'s and moves held-out probabilities by 0.0017.

## svy covers descriptive statistics only, for now

The replication's cross-validation is unweighted by design, matching the original analysis, so `svy` is needed only for design-based descriptive numbers.

It is worth being precise about how unweighted the original is. `Selected_Child_Weight` sits in the design matrix as an ordinary predictor, but R's lasso zeroes it in all 60 splits. The published analysis therefore does not use the survey weight in any sense, as a weight or as a feature. `Unique_Household_ID` is likewise present in the design and dropped in all 60 splits, which settles the household-leak question more firmly than a performance check could.

The survey-weighted extension will lean on `svy` much harder. Its installability is an open question tracked in the planning docs, and it stays out of `pyproject.toml` until that's settled.

## Care coordination is a predictor, not an outcome

The extension predicts three outcomes: foregone care (K4Q27), emergency department use (HOSPITALER), and behaviour therapy (AUTISMTREAT), with the autism group defined as children ever diagnosed (K2Q35A).

Care coordination was considered as a fourth. Olivia ruled it out as an outcome while keeping it as a predictor the project cares about interpreting. Nothing predicts it. It needs to survive the pipeline as a feature, and it should appear by name in the coefficient rankings and the category ablation rather than being left to turn up incidentally.

Two things follow that nobody has checked. Whether the relevant variables reach the design matrix under the current config, and whether R's lasso retains any of them. `analyses/inspect_r_coefficients.py` answers the second once they are added to its watch list, which is how the household identifier and the survey weight were settled.

## Fixtures are fetched, not committed

The NSCH_autism splitter-validation fixture is 46,010 rows by 364 columns, too heavy for the repo. Tests that need it download it from the Zenodo record (10.5281/zenodo.18273949) with a pinned checksum, cache it under `fixtures/cache/` (gitignored), and skip when offline (`network` marker). The one-line golden header from the R model matrix is tiny and is committed.

## Fold assignment: replicate the rules, import the draws

R's `sample()` under `set.seed()` and NumPy's generator are different RNGs, so no seed makes a Python fold assignment or downsample reproduce R's draw for draw. The splitter therefore splits equivalence in two. The deterministic part, the mapping from a fold assignment to same/other/all index sets, is pure set logic and matches mlr3resampling exactly; the equivalence tests feed the fold column exported from the R analysis through `assign_folds(precomputed=...)` (mirroring mlr3resampling's own user-supplied fold role) and assert the resulting index sets full-vector. The random parts replicate R's rules, not its draws: fold assignment shuffles and deals each (subset, outcome) cell round-robin, and the `sizes=0` downsample transcribes the R source's nominal-size arithmetic (`same = floor(full * (K-1)/K)`, `all = sum(same)`, `other = all - same`; per-stratum kept counts of `floor(L_s * target / nominal_own)`). Each downsampled split draws from an independent stream derived from the seed and the split's identity, so results never depend on iteration order or parallel scheduling.

A structural detail that follows from the same set logic: for any fold, "same" on 2019 and "other" on 2020 train on identical rows. Their fitted coefficients are bit-identical, so the 60 full splits hold 30 distinct models. Anything averaging across splits should know that.

## The R runs do not all partition the data the same way

Confirm that two runs share a fold assignment before comparing them fold by fold. This sounds obvious and cost us most of a week.

Ten R runs of the same analysis fall into three groups when paired by fold number: five February and March runs, four August runs, and batchtools alone. Within a group the mean absolute difference in per-fold AUC is about 0.0005. Between groups it is about 0.0073, which is larger than the effects the study reports, and we spent several days believing that meant something about the analysis.

It does not. Mean AUC within each of the six cells, which does not depend on how folds are numbered, shows no separation at all: the largest gap across all 55 pairs is 0.00051. And optimal matching, which finds the fold pairing that minimises the distance, closes 64 to 66 percent of the between-group gap while closing only 5 to 23 percent within a group. Matching always helps a little by chance; three times as much, with no overlap across 44 pairs, means the folds are the same children under different numbering.

The draw belongs to `mlr3resampling`. `NSCH.R` calls `set.seed(1)` then `SOAK$instantiate(task.obj)`, and `mlr3learners` plays no part in it. What changed the draw between February and August is untested, and no February artifact records its assignment, so this stays an inference from the AUC values.

Batchtools sits apart from its own February siblings by the same measure, which is what dispatching jobs to separate processes does to a draw that depends on the order the random stream is consumed.

Our fixture, `nsch_autism_folds.csv`, was drawn by `mlr3resampling` 2026.5.19 under seed 1; the provenance file beside it records that. Our splitter reads it, so our folds match the August runs. That has been confirmed directly against `NSCH_seed1` rather than assumed: joining held-out predictions on split and row identifier matched all 138,030 rows with the outcome agreeing on every one.

The published results are in `data_Classif_batchmark_registry.csv` in the cv-same-other-paper repository and reproduce the figures in Section 4.3 to four decimals. They used `same_other_cv`, the class since removed, so their folds correspond to nothing we have and only cell means are comparable. On those they sit in the middle of the pack.

One further version caution. `ResamplingSameOtherCV`, which the original analysis used, is gone from current mlr3resampling, present in the 2024.9.6 archive and absent by 2026.5.19. Any R-side rerun needs the package version pinned.

## Three label conventions, none visible in an AUC

The reference pipeline carries three separate conventions about which class is positive. They compose correctly, which is luck rather than design, and each would independently invert a probability-scale comparison while leaving every AUC untouched.

The design matrix's outcome column `y` holds the strings `"Yes"` and `"No"`, not 1 and 0. mlr3 assigned `"No"` as the positive class, so R's saved probabilities are one minus the probability of the outcome, and glmnet's coefficients point the same way; a reconstructed linear predictor reproduces exactly `1 - AUC` until it is negated. scikit-learn's `roc_auc_score`, handed string labels, treats the lexicographically last as positive, which is `"Yes"`.

Any code touching probabilities or coefficient signs has to account for all three. `analyses/repair_r_predictions.py` documents the composition and checks it two ways: the recovered probabilities must reproduce R's own AUC, and their mean must land on the 3.05% outcome prevalence rather than its complement.

## R's fitted models are recoverable without refitting

`LearnerClassifCVGlmnetSave` stores the fitted coefficients at the selected lambda, which makes coefficient and ranking comparisons possible from archived runs alone. `analyses/verify_r_coefficients.py` confirms the recovery by reconstructing R's reported AUC from the weights, matching on all 60 splits to 1e-14.

Two defaults get in the way and both are one argument each. The package's `save_learner_glmnet` drops the intercept (`coef(x$model)[-1, ]`), which is sensible for interpretation and useless for prediction; passing a custom `save_learner` keeps all 365 rows. And `proj_grid` takes `save_pred`, which defaults to `FALSE`, so predictions are discarded unless asked for. Twenty-five minutes of compute was spent discovering the second one.
