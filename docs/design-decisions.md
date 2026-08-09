# Design decisions

Why the package is built the way it is. Each entry records a decision a
maintainer might otherwise re-litigate. Longer treatments live in
`planning/workstream-c-scoping.md` and in the documents named below; these are
the short versions.

## Equivalence is judged on predictions, not coefficients

glmnet and scikit-learn standardize inputs differently and parameterize
penalty strength differently, so coefficient-level agreement between the R
analysis and this port was never going to be reachable. That was the a priori
argument. There is now a measurement behind it.

Three R runs differing only in the `cv.glmnet` inner seed, same machine, same
afternoon, same package build, disagree with each other by 0.004 mean absolute
on the fitted weights, by as much as 0.377 on a single weight where the
largest weights run to about 1.04, and on which features are selected at all
for roughly two percent of the 364. The reference does not have coefficient
stability, so requiring it of a reimplementation would be measuring the
lasso's behaviour and calling it our error.

The replication contract is held-out predicted probabilities and score
rankings against margins fixed in advance. Those margins, the R-internal
yardstick they are anchored to, and what is gated versus merely reported are
in `docs/equivalence-margin.md`. What differs between the two implementations
and why is in `docs/replication-equivalence.md`.

## We wrote our own SOAK splitter

soakpy (PyPI 0.0.54) stratifies on the subset only rather than the (subset,
outcome) pair, its downsampling isn't seeded, and its high-level API is
regression-only. Our splitter stratifies on (subset, outcome), seeds the
downsample so the fairness runs are reproducible at both size settings, and
returns index arrays that plug into scikit-learn and XGBoost directly. soakpy
remains a cross-check on the plain partitioning, nothing more.

## The linear learner is penalized logistic regression, and the penalty is lasso

The original analysis fits binomial `cv_glmnet` models. The faithful
scikit-learn analogue is `LogisticRegression` with an explicit penalty, not
`Ridge` or `RidgeClassifier`, which minimize squared error on the 0/1 labels
and would quietly break a prediction-level comparison.

An earlier version of this entry said the core analysis used ridge and that
lasso appeared only in the fairness driver. That was wrong. `cv.glmnet`
defaults to `alpha = 1`, and R's saved coefficients confirm it: 65 of 364
weights are nonzero on a representative split, 255 features are never selected
in any of the 60 splits, and 15 are selected in all of them. The core analysis
is lasso, and a ridge port compared against it tests a choice we made rather
than the fidelity of the port.

The `C`-to-lambda mapping and the standardization handling are documented
alongside the learner wrappers. One consequence worth knowing before running
anything: scikit-learn needs the `saga` solver for L1, which took eleven and a
half hours on the 60 splits where ridge with `lbfgs` took eight minutes.
glmnet's coordinate descent handles the same problem in minutes.

## svy covers descriptive statistics only, for now

The replication's cross-validation is unweighted by design, matching the
original analysis, so `svy` is needed only for design-based descriptive
numbers.

It is worth being precise about how unweighted the original is.
`Selected_Child_Weight` sits in the design matrix as an ordinary predictor,
but R's lasso zeroes it in all 60 splits. The published analysis therefore
does not use the survey weight in any sense, as a weight or as a feature.
`Unique_Household_ID` is likewise present in the design and dropped in all 60
splits, which settles the household-leak question more firmly than a
performance check could.

The survey-weighted extension will lean on `svy` much harder. Its
installability is an open question tracked in the planning docs, and it stays
out of `pyproject.toml` until that's settled.

## Fixtures are fetched, not committed

The NSCH_autism splitter-validation fixture is 46,010 rows by 364 columns, too
heavy for the repo. Tests that need it download it from the Zenodo record
(10.5281/zenodo.18273949) with a pinned checksum, cache it under
`fixtures/cache/` (gitignored), and skip when offline (`network` marker). The
one-line golden header from the R model matrix is tiny and is committed.

## Fold assignment: replicate the rules, import the draws

R's `sample()` under `set.seed()` and NumPy's generator are different RNGs, so
no seed makes a Python fold assignment or downsample reproduce R's draw for
draw. The splitter therefore splits equivalence in two. The deterministic
part, the mapping from a fold assignment to same/other/all index sets, is pure
set logic and matches mlr3resampling exactly; the equivalence tests feed the
fold column exported from the R analysis through `assign_folds(precomputed=...)`
(mirroring mlr3resampling's own user-supplied fold role) and assert the
resulting index sets full-vector. The random parts replicate R's rules, not
its draws: fold assignment shuffles and deals each (subset, outcome) cell
round-robin, and the `sizes=0` downsample transcribes the R source's
nominal-size arithmetic (`same = floor(full * (K-1)/K)`, `all = sum(same)`,
`other = all - same`; per-stratum kept counts of
`floor(L_s * target / nominal_own)`). Each downsampled split draws from an
independent stream derived from the seed and the split's identity, so results
never depend on iteration order or parallel scheduling.

A structural detail that follows from the same set logic: for any fold, "same"
on 2019 and "other" on 2020 train on identical rows. Their fitted coefficients
are bit-identical, so the 60 full splits hold 30 distinct models. Anything
averaging across splits should know that.

## The R reference is not a fixed point

Pin the package build before quoting any tolerance. Ten R runs of the same
analysis fall into groups by mean absolute AUC distance: runs sharing a build
agree to about 0.0005, and runs on different `mlr3learners` builds differ by
about 0.0073. That is larger than the effects the study reports, and larger
than the distance between our port and the current build.

The `cv.glmnet` inner seed is worth about 0.0005, the same as running on a
different machine. Leaving that seed unset is bit-identical to setting it to 1,
because the learner defaults to 1, so the February runs were effectively seeded
all along.

There are at least four independent sources of randomness in a reference run
and none is obvious from the call site. `NSCH.R` calls `set.seed(1)` before
instantiating the resampling, which fixes the outer folds. The learner carries
its own inner cross-validation seed. `proj_grid` exposes `train_seed` and
`resampling_seed` on top of both. Only the learner seed has been varied
deliberately; the others are untested.

One further version caution: the class the original analysis used,
`ResamplingSameOtherCV`, has been removed from current mlr3resampling (present
in the 2024.9.6 archive, gone by 2026.5.19). The semantics here were pinned
against the archived source, and any R-side rerun needs the package version
pinned accordingly.

## Three label conventions, none visible in an AUC

The reference pipeline carries three separate conventions about which class is
positive. They compose correctly, which is luck rather than design, and each
would independently invert a probability-scale comparison while leaving every
AUC untouched.

The design matrix's outcome column `y` holds the strings `"Yes"` and `"No"`,
not 1 and 0. mlr3 assigned `"No"` as the positive class, so R's saved
probabilities are one minus the probability of the outcome, and glmnet's
coefficients point the same way; a reconstructed linear predictor reproduces
exactly `1 - AUC` until it is negated. scikit-learn's `roc_auc_score`, handed
string labels, treats the lexicographically last as positive, which is
`"Yes"`.

Any code touching probabilities or coefficient signs has to account for all
three. `analyses/repair_r_predictions.py` documents the composition and checks
it two ways: the recovered probabilities must reproduce R's own AUC, and their
mean must land on the 3.05% outcome prevalence rather than its complement.

## R's fitted models are recoverable without refitting

`LearnerClassifCVGlmnetSave` stores the fitted coefficients at the selected
lambda, which makes coefficient and ranking comparisons possible from archived
runs alone. `analyses/verify_r_coefficients.py` confirms the recovery by
reconstructing R's reported AUC from the weights, matching on all 60 splits to
1e-14.

Two defaults get in the way and both are one argument each. The package's
`save_learner_glmnet` drops the intercept (`coef(x$model)[-1, ]`), which is
sensible for interpretation and useless for prediction; passing a custom
`save_learner` keeps all 365 rows. And `proj_grid` takes `save_pred`, which
defaults to `FALSE`, so predictions are discarded unless asked for. Twenty-five
minutes of compute was spent discovering the second one.
