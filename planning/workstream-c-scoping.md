# Workstream C: replication scoping, and the survey-weighted extension

*Scoping the SOAK replication and the Q1 + Q4 papers it sets up.*

Chris Reger · ASD3 Outcomes · June 2026 · Draft for team review

**Status:** working draft. **Reviewers:** Ben Lucas and David Folch; Toby Hocking on the SOAK extension (§5). Source code checked against the vas235 repos 2026-06-25; links and references verified 2026-07-07.

If you're reading this for review: §1–§4 are mostly settled and mechanical. Spend your time on §5 and §7 instead. The question that matters most for the timeline is whether the Q1 uncertainty work is tractable on my schedule or needs hands-on help from Ben or Toby. And if you only double-check one technical detail, make it the one-hot reference rule in §4. The decisions list in §7 tracks what's settled and what's open.

## Executive summary

Workstream C ports the published 2016–2023 ASD analysis from R to Python, then checks the port against the existing R results by comparing predictions rather than coefficients. At the center of it is SOAK, short for Same/Other/All K-fold cross-validation. Odd name, simple idea. Hold out some children from one time period as a test set. Train three models: one on the rest of that same period, one on the other periods, one on everything. If the "other" model predicts the held-out children about as well as the "same" model does, the periods look statistically interchangeable and pooling them is defensible. If it does worse, something changed.

The replication isn't the end in itself. It's the base for the two papers I want to lead with: Q1, a survey-weighted version of SOAK (the method), and Q4, an interpretable, survey-correct look at unmet service need for children with autism (the payoff). Both run off the same harmonized dataset and mostly the same code, so the replication, the toolkit, and the extension amount to one line of work. This doc scopes the replication in detail (§1–§4), lays out the extension and what new work it needs (§5), and ends with the decisions I want the team's read on (§7). The fuller case for the paper direction lives in the Candidate Research Questions memo.

At a glance:

- The arc: faithful replication first, then a survey-weighted SOAK (Q1) and a survey-correct unmet-need analysis (Q4), off one dataset and codebase.
- One new repo, `nsch-ml-py`, with our own SOAK splitter as an internal module.
- Highest replication risk: the conditional one-hot reference rule (§4). Get a single reference level wrong and every model's baseline shifts, with no error message anywhere.
- Biggest new build: a design-aware SOAK with honest population-level uncertainty (§5). Not in the current queue. This is where the novel methods work lives.
- Settled: 2024 goes into P5 (Olivia's call).
- Open and gating Phase 1: the prediction-tolerance metric (§7).
- Next step: stand up the repo skeleton and pin the one-hot encoder against the real column header as a golden test.

## 1. Phases

The work is paced by the HS 685 plan, which splits the replication three ways. Phase 1 (weeks 1–5) is descriptive statistics and the linear model. Phase 2 (weeks 5–9) includes XGBoost and SHAP, plus the two analyses that build on them: SHAP clustering and the fairness SOAK. Phase 3 (weeks 9–11) re-runs everything on 2016–2024 and is the stretch. The survey-weighted work in §5 comes after all three.

One wording flag before the table, because the plan is loose here and it tripped me up. Both the plan and the original analysis fit penalized logistic regression through glmnet. glmnet has a single knob, `alpha`, that picks the flavor of penalty: `alpha = 0` is ridge (shrinks every coefficient toward zero, keeps them all), `alpha = 1` is lasso (can zero coefficients out and drop variables entirely). The plan calls Phase 1 "elastic-net," the in-between setting. Neither driver actually uses it. The [core driver](https://github.com/vas235/ASD3-machine-learning/blob/main/ml_to_registry.R) sets `alpha = 0.0`, so ridge; the [fairness driver](https://github.com/vas235/ASD3-machine-learning/blob/main/2016_2023_fairness_SOAK/ml_to_registry.R) never sets `alpha` at all, leaving glmnet at its default of 1, lasso. Sub-analyses A and D run different linear models. The port has to match each one separately.

The four sub-analyses land like this:

| Sub-analysis | Phase | What lands |
|---|---|---|
| A, linear part | 1 | Descriptive tables; ridge full-task SOAK across periods; the `only_<cat>` and `not_<cat>` category tasks; ridge coefficients at `lambda.min` (top-10); the cheap baselines that share the harness and need no SHAP |
| A, boosted part | 2 | XGBoost full-task SOAK; ROC with 0.5 points and the high-specificity zoom; same/other/all accuracy by period; confusion matrices at the FNR targets |
| B (SHAP) | 2 | Exact TreeSHAP on the full-task XGBoost models |
| C (SHAP clustering) | 2 | k-means over per-child SHAP vectors; CH/DB model selection; the k=4 seed=4 characterization |
| D (fairness) | 2 | glmnet-only SOAK, one fairness subset at a time, at `sizes=-1` and `sizes=0` |
| Year extension | 3 | Re-run the chain on 2016–2024; characterize anything specific to 2024 |

### Phase 1

Descriptive tables come first, and they're mechanical: read the same data, compute the same summaries, match the published format. The linear SOAK is where the choices start. On the R side it's the glmnet driver in the analysis repo, `cv_glmnet` at `alpha = 0`, run through [`mlr3resampling::ResamplingSameOtherCV`](https://github.com/tdhock/mlr3resampling). Two extra sets of runs ride alongside the full model: `only_<cat>` tasks that fit on a single category of survey questions at a time, and `not_<cat>` tasks that fit on everything except that category. Together they show how much each question category contributes. For how the mlr3resampling pieces fit together, the cleanest worked example is the [NSCH-autism reproduction code](https://github.com/tdhock/cv-same-other-paper) in cv-same-other-paper. On the Python side, I'd do the data prep in Polars and cross over to pandas at the sklearn boundary, the same hybrid `nsch-py` uses.

The linear model needs care. glmnet and sklearn won't produce the same coefficients even on identical data, because they standardize the inputs differently and they parameterize the penalty strength differently, so the equivalence check runs on held-out predicted probabilities instead. The faithful sklearn analog is L2-penalized logistic regression, `LogisticRegression(penalty='l2')`, since `cv_glmnet` here is fitting a binomial model. It is not `Ridge` or `RidgeClassifier`. Those minimize squared error on the 0/1 labels, a different objective that would quietly break the comparison, and even with the right estimator, the mapping between sklearn's `C` and glmnet's lambda takes some algebra. What about `python-glmnet`, which tracks glmnet's lambda path more closely? More viable than I first thought. The original repo was archived in mid-2024. Still, a maintained fork publishes wheels on PyPI (2.6.1) for Python 3.10 through 3.13, so it installs cleanly across our CI matrix. Upstream is dead and the fork is thin, though, so I'd pin sklearn as the primary and use `python-glmnet` as a cross-check. Design-based descriptive numbers go through [`svy`](https://svylab.com/docs/svy). The splitter is the one we write ourselves, described in §2. The featureless, rpart, and kNN baselines ride along in this phase, since they share the harness and don't touch SHAP.

When is Phase 1 done? When the descriptive tables agree within rounding, the ridge held-out probabilities agree within whatever tolerance we settle on, and AUC and accuracy match per period and per same/other/all. A quick word on AUC for the non-modelers: it measures how well the model ranks. A score of 1.0 means every child with ASD scores above every child without; 0.5 is a coin flip. It's the better metric here because only about 3% of children in the data carry an ASD diagnosis, and at that prevalence plain accuracy rewards a model that just says "no" to everyone. One more criterion, and it matters: SOAK's headline result is the per-fold error difference between same, other, and all, with its p-value from a two-sided paired t-test on K−1 degrees of freedom. Reproducing that comparison goes on the success list too. The ridge top-10 coefficients at `lambda.min` (the penalty strength that minimized cross-validated error) should line up on sign and rough ordering; exact magnitudes won't match, because of the standardization gap.

### Phase 2

XGBoost should be the easier half, since the Python and R packages share the same C++ core. Should be. The agreement holds only if the tree method, the `base_score` starting point, and the missing-value handling all match, and if the tuned settings reproduce, which requires the inner cross-validation folds and the tuning selection to line up. A mismatch here is more likely a configuration gap than a bug in the port. The tuning searches `eta` (the learning rate) on a log grid over [0.001, 1] and `nrounds` (the number of trees) over [1, 100], grid resolution 5, scored on AUC, with a 5-fold inner CV (`ResamplingIgnoreGroupCV`) that ignores the period structure.

SHAP is how we ask the boosted model to explain itself. It splits each child's predicted probability into additive contributions from each input, so you can say which survey answers pushed a given child's prediction up or down. The R analysis uses exact TreeSHAP via [`predict(..., predcontrib = TRUE, approxcontrib = FALSE)`](https://github.com/vas235/ASD3-machine-learning/blob/main/get_shap.R). The values are computed out of fold, on the held-out rows of each split, so the model never explains children it trained on. They're averaged per child, checked for additivity at 1e-6 (the contributions must re-sum to the prediction), then rolled up from one-hot columns to the underlying survey question for the top-10. Python's `Booster.predict(pred_contribs=True)` computes the same quantity. `shap.TreeExplainer` is the alternative.

The clustering asks a follow-up question: do children fall into groups whose predictions were driven by similar answers? k-means on the per-child SHAP vectors. Z-score the columns, optionally normalize each child's vector to unit length, sweep k from 2 to 13 across random seeds 1 to 10 with `nstart` 5, and score each result on Calinski-Harabasz and Davies-Bouldin, two standard indices of how well-separated the clusters are. The published characterization is at k=4, seed 4. sklearn covers all of it: `KMeans` with `n_init=5`, `StandardScaler`, `Normalizer`, both scoring metrics.

The fairness SOAK reuses the same machinery with a different question. Instead of asking whether time periods are interchangeable, it asks whether demographic groups are, one grouping at a time: race, poverty level, sex, household language, insurance type. glmnet only. Each grouping variable's own one-hot columns get deleted from the features first, so the model can't peek at the group label. Every run happens twice, at `sizes=-1` (full training sets) and at `sizes=0` (training sets downsampled to the smallest group's size, so a group's result isn't just an artifact of having more data).

Phase 2 is done when the XGBoost held-out predictions agree within a tighter tolerance than Phase 1, the same/other/all comparison and its p-value reproduce as in Phase 1, the SHAP top-10 rolled up to survey questions matches, the clustering picks the same k with agreeing score curves and reproduces the k=4 seed=4 characterization in both raw and winsorized form, and the fairness per-group same/other/all numbers agree at both sizes. Confusion matrices get checked at the published false-negative-rate targets of 0.1, 0.2, and 0.5. (False-negative rate: the share of children with ASD the model misses at a given decision threshold.)

### Phase 3

Same drivers, pointed at 2016–2024. This is the only phase that needs Workstreams A and B finished, since it consumes the cached R 2016–2024 output and the validated Python merge. 2024 goes into P5, per Olivia, so it sits in its own two-year bin until 2025 data lands.

One caution carries in from §5, and I had the mechanism wrong until I read the Census documentation. The weighting story is not that 2024 is a special wave. The Census Bureau revised the race and ethnicity imputation and weighting starting with the 2022 NSCH and applied it backward: revised 2021 files re-released October 2023, revised 2016–2020 files April 2024. MCHB says the revised files are the ones to use for any combining or comparing across years, and the non-comparability warnings floating around apply to estimates computed from the pre-revision files (including CAHMI's archived 2016–2021 queries) versus the revised ones. Practical consequence for us: the multi-year work has to sit on the revised files throughout. That adds a check to this phase. We need to confirm the raw .dta files behind our harmonized dataset were downloaded after the April 2024 re-release, because anything weighted, from the `fwc` descriptives to all of §5, depends on it.

Phase 3 is done when the pipeline runs end to end on the extended data and anything specific to 2024 is written up. If A or B slip, this phase compresses first and then becomes future work the paper notes.

## 2. The SOAK splitter

I'd write our own small splitter rather than depend on soakpy. Here's the job, using our data as the example. Every child belongs to a two-year period, P1 through P4. Assign each child a fold number from 1 to 10, dealt out so that every combination of period and outcome (ASD yes/no) spreads evenly across the folds. That's the stratification, and it matters: roughly 3 in 100 children have the diagnosis, so an unstratified deal could leave a fold with almost no positive cases. Then, to test on period P4 with fold 3 held out:

- Test: fold 3 within P4
- Same: folds 1–2 and 4–10 within P4
- Other: folds 1–2 and 4–10 within P1, P2, and P3
- All: folds 1–2 and 4–10 everywhere

Cycle that over every period and fold and you get the full grid of same/other/all comparisons. Grouping falls out automatically here, because the group is the period itself. `sizes=-1` runs the training sets at full size. `sizes=0` adds a variant where every training set is downsampled to the smallest group's size, which the fairness runs use, and that downsample has to be seeded or the two settings can't be compared reproducibly. The inner tuning CV is a separate, simpler splitter, a plain stratified 5-fold that ignores period and group, used inside the XGBoost and kNN hyperparameter searches. Reference implementations: [`mlr3resampling`](https://github.com/tdhock/mlr3resampling) and the [SOAK paper](https://doi.org/10.1002/sam.70055) (Hocking et al., SADM 2026; Olivia is a co-author). Writing the splitter ourselves also leaves room to add the design-aware folds the extension needs (§5) without fighting someone else's API.

Why not soakpy? Four reasons, and it stays a reference (PyPI, version 0.0.54) rather than a dependency. It stratifies on the subset only, not on the (subset, outcome) pair, which matters with an outcome this imbalanced; in the paper's NSCH dataset the majority class outnumbers the minority 31.8 to 1. Its downsampling isn't seeded, and the fairness runs need it to be. Its high-level API is regression-only, and this is classification. And it pulls in more dependencies than the package should carry for one splitter. Caveat on my own claims: I haven't been through its 0.0.54 source line by line, so confirm those specifics before leaning on a soakpy comparison.

Validation happens two ways. First, against the R fold assignments on the public NSCH_autism.csv fixture (the SOAK datasets live on [Zenodo](https://doi.org/10.5281/zenodo.18273949); [cv-same-other-paper](https://github.com/tdhock/cv-same-other-paper) builds them). Same seed, folds=10, then assert that the fold IDs and the same/other/all index sets match exactly for every subset and fold. Second, against `soakpy.split` on the plain partitioning, where the two should agree, as a cheap way to catch logic bugs. The fixture's subsets are survey years (2019 with 18,202 children, 2020 with 27,808) while ours are periods, but the splitter doesn't care what the subset means, so the fixture still exercises the logic. One detail worth carrying through: the original analysis used different random seeds in different places. Core benchmark, seed 1. Fairness runs, seed 42. The clustering swept seeds 1 to 10 and reported seed 4. Reproducing the study's actual folds and cluster labels means matching those, so the splitter's seed handling has to make that possible per analysis.

## 3. Repository

One repo, `nsch-ml-py`, next to [`nsch`](https://github.com/NAU-ASD3/nsch), [`nsch-py`](https://github.com/NAU-ASD3/nsch-py), and [`nsch-ml-paper`](https://github.com/NAU-ASD3/nsch-ml-paper), with the splitter as an internal module. Scaffolding mirrors `nsch-py`: uv, ruff, mypy strict, pytest, mkdocs-material, the same CI matrix over Python 3.11/3.12/3.13. `CONTRIBUTING.md`, `design-decisions.md`, and `CODEOWNERS` carry over, with Ben and David as the reviewers. Stacked PRs in dependency order. Version bumps in the `2026.M.DD` format, with a `CHANGELOG` entry each.

Rough layout:

```
src/nsch_ml/
  data/      prep: metro_yn imputation, the >10% drop and complete-case, race7/povlev4/fairness copies, period bins, conditional one-hot
  soak/      the splitter and the inner ignore-group CV
  learners/  thin wrappers: ridge, lasso, xgboost, knn, rpart-equivalent, featureless
  shap/      TreeSHAP extraction, OOF averaging, base-variable collapse
  cluster/   k-means over SHAP vectors, CH/DB, the characterization
  fairness/  the fairness-subset driver
  report/    ROC and the high-specificity zoom, confusion matrices at the FNR targets, coefficient tables
tests/       same layout, plain asserts, synthetic data, full-vector
docs/        design-decisions, splitter notes, the add-a-year runbook
fixtures/    NSCH_autism.csv for the splitter tests, plus the SOAK CSV header as the one-hot golden
```

This repo consumes `nsch-py`'s `get_clean_data()` output. The R [`ml-final-prep.R`](https://github.com/vas235/ASD3-machine-learning/blob/main/ml-final-prep.R), which handles the missing data, the one-hot encoding, and the final SOAK CSVs, becomes the `data/` module here. It doesn't get its own repo. That prep is specific to this analysis, the same boundary we already drew for the DevScrnng join: study-specific transforms live with the analysis, general harmonization lives with the harmonizer. The design-aware resampling for §5 lands in `soak/` later, so the layout already has a home for it. `design-decisions.md` should record the reasoning a maintainer will want later: why the one-hot rule has to match exactly, why equivalence is judged on predictions, why we wrote our own splitter, and why `svy` covers only the descriptive numbers while the replication CV runs unweighted.

## 4. Constraints

Equivalence is judged on predictions, not coefficients (plan §5, §8). For the linear models that's forced by the glmnet/sklearn differences described in Phase 1. XGBoost can hold a tighter tolerance once the configuration matches, since the two languages share the same core. SHAP is exact and the `shap` package is the reference implementation, so the top-10 questions is the qualitative check there.

The riskiest single detail in the whole replication is the one-hot encoding rule. Worth spelling out what that means. One-hot encoding turns a categorical survey answer into 0/1 columns. Take a three-level variable like `higrade`, parental education: less than high school, high school, more than high school. The encoder creates columns for two of the levels and leaves the third out. The left-out level is the reference. The model measures every effect relative to it, so the "high school" coefficient means "compared to the reference." Now choose a different reference. Every coefficient in the model shifts, the predictions stay almost identical, and nothing errors out. That's the failure mode to fear in a replication: quietly measuring against a different baseline while everything appears to work.

The actual rule in [`ml-final-prep.R`](https://github.com/vas235/ASD3-machine-learning/blob/main/ml-final-prep.R) is conditional, and it isn't the rule I'd assumed before reading the code. The line is `drop_first <- if (n_levels == 2) (col %in% reverse_set) else !(col %in% reverse_set)`, which then encodes `levels[-1]` when `drop_first` is true (first level becomes the reference) or `levels[-length]` otherwise (last level becomes the reference). Worked through: a variable outside the reversed set drops its first level if it has three or more levels, and its last level if it has two. The sixteen reversed variables flip both of those. The reversed set is `sc_sex`, `birthwt`, `k4q22_r`, `k4q24_r`, `wgtconc`, `k4q04_r`, `k5q20_r`, `k5q31_r`, `higrade`, `k5q11`, `instype`, `k8q11`, `higrade_tvis`, `arrangehc`, `athomehc`, `k4q02_r`.

A few variables sit outside the encoding entirely. race7, povlev4, and the fairness copies stay categorical because they serve as SOAK subsets. The survey weight `fwc` is dropped from the features along with `period`, `stratum`, `hhid`, `state`, `fipsst`, and `year`; dropping `fwc` is what makes the replication's model fitting unweighted by design, and §5 is about what that costs. The headline feature count is 288. The task in the registry is literally named `all__all.288`.

I'd test this head on rather than trust a careful read: pull the column header the R pipeline actually feeds the learner, and assert the exact one-hot column set and the reference level per variable. Pin the columns, not the count.

Two smaller wrinkles from the same script. The age-at-diagnosis column `k2q35a_1_years` is exempted from the missing-data filters, written out to `asd_diagnosis_age.csv` row-aligned, then dropped from the main matrix; the clustering later reads age back from that side file. And the `metro_yn` imputation feeding the prep comes from [`vas235/MSA-imputation`](https://github.com/vas235/MSA-imputation).

On `svy`. In the replication it's only for the design-based descriptive numbers, since the CV is unweighted, but confirming it installs matters more than that suggests, because the extension leans on it for actual inference. The plan points at [svylab.com/docs/svy](https://svylab.com/docs/svy) and cites 0.15.0; last I checked, the repo wasn't downloadable yet. If it won't install, the descriptive numbers fall back to weighted summaries computed in Polars, with design-based standard errors done by hand where the published tables need them. The extension is the harder case. Design-based inference is the whole point of Q1, so if svy stalls there, the backstop is calling R's survey package through rpy2, or implementing the Taylor-linearization variance for NSCH's stratified design directly. NSCH ships no replicate weights, so replication-based variance isn't available unless we construct our own replicates. Either way, Q1 needs an answer on svy before it starts.

Monsoon stays as is. The SLURM scripts call into the Python package the same way they call into R now, so the job-array structure doesn't change; a Python driver wraps the splitter where batchtools wrapped it before. Pins for the analysis stack: scikit-learn (which also covers the clustering), `xgboost`, `shap`, `polars`, `python-glmnet` as the cross-check, and `svy` if it installs.

On process: stacked PRs in dependency order, one branch per PR, the full CI run green locally before I ask for review, and a written plan before anything that touches more than a couple of files. Ben and David review. The test style is what we settled on in the R repo and carried into `nsch-py`: plain asserts, full-vector comparisons, synthetic data, one behavior per test.

## 5. The survey-weighted extension (Q1 + Q4)

Start with why the weights exist. NSCH doesn't sample households with equal probability, so each surveyed child stands in for a different number of children in the U.S. population, and the weight (`fwc`) says how many. Ignore the weights and you've described your sample. Use them properly and you can talk about American children. The replication reproduces an analysis that drops the weights from model fitting, which is the right call for a faithful port and the wrong call for a population claim, and closing that gap is what the two lead papers are built on. Q1 is the method, a survey-weighted SOAK. Q4 is the demonstration that makes the method matter to the families the grant is for, an interpretable, survey-correct look at unmet service need for children with autism. They share the harmonized dataset and most of the codebase with the replication, so this extends the same line of work. The fuller case, venues and the candidate questions I'm not leading with included, is in the Candidate Research Questions memo. Here I'm scoping the work and the decisions.

There's a tempting shortcut, and it isn't enough. scikit-learn and XGBoost both accept per-row sample weights, so you can hand them `fwc` and the point predictions come out about right. The uncertainty doesn't. Sample weights treat the data as if it were a simple random sample that happens to have importance factors attached, which understates the variance the actual sampling design introduces, so confidence intervals come out too narrow and p-values too small. Honest population-level uncertainty needs design-based methods, the survey-statistics machinery built for exactly this. The week-1 toy-data check already in the queue is meant to confirm the shortcut reproduces the original point predictions. Necessary, and only half the story. What makes the design-based path workable in Python now is [`svy`](https://svylab.com/docs/svy), Diallo's successor to samplics, which its authors validate against R's long-standing survey package.

Q1 is a design-aware SOAK, and since part of the territory is already mapped, the novelty has to be stated precisely. Wieczorek, Guerin and McMahon's surveyCV (2022) already does cross-validation that respects a sampling design: its folds honor the strata and keep sampling clusters intact so information can't leak between train and test, and its test error is estimated with design-based methods. What doesn't exist yet is a survey-aware version of SOAK's particular comparison, valid inference on whether training on "other" or "all" beats training on "same," covering both the error differences and their p-values, under the design. Extending design-based CV to that subset-comparison question is the contribution. surveyCV means we start from a foundation rather than a blank page.

I chased down NSCH's design specifics, because they decide what "design-respecting" means for us. Per the Census FAQs and Analytic Guide, the public files ship exactly three design variables. FIPSST and STRATUM together form the strata (state crossed with a 2-level within-state stratum). HHID is the primary sampling unit, meaning each household is its own sampling unit. There are no replicate weights. Two consequences follow. First, cluster-level folds reduce to household-level folds for us, since one child's records never span households, so the fold construction is stratified sampling over households, which our splitter nearly does already. Second, variance has to come from Taylor linearization, the standard survey-statistics route to standard errors when you have strata and weights but no replicate weights, or from replicates we construct ourselves. Extending that variance machinery to a *difference* in cross-validated error between same/other/all training sources is the part nobody has done. That's the open methods question, and it's the one to put in front of Ben and Toby. A simulation study comes with it. To claim the weighted version's confidence intervals are valid, we show they cover the truth at the stated rate; that's standard for a methods paper, and it isn't in the queue.

Q4 changes the prediction target. The replication predicts ASD diagnosis itself. Q4 predicts unmet health-care need and weak care coordination among children with autism across 2016–2024, then asks which child, family, and system factors drive them and whether the disparities have moved. Three pieces are new analysis rather than replication: defining the unmet-need and care-coordination outcomes so they mean the same thing across nine years of changing questionnaire items; fitting a survey-correct interpretable model, the same penalized-regression-and-SHAP machinery the replication builds but weighted; and breaking results out by race and ethnicity, rurality, and insurance, with a check that the weighting revision isn't driving any apparent trend.

Now the honest accounting of what's planned versus what's new, because this is the part that bites the timeline. Already in the queue: the harmonized 2016–2024 dataset and Python pipeline (Workstreams A and B), the replication itself (Phases 1–2 here), the 2024 wave (Phase 3), and the toy-data check on the sample-weights shortcut. Not yet in the queue, and needed for the papers: the design-aware SOAK method, the coverage simulation, Q4's outcomes defined across the full span, the disparities breakdown and the weighting sensitivity check, design-correct resampling built into the pipeline rather than only verified on toy data, and the writing. The replication gets us a working pipeline and a validated baseline. That's most of the software paper and the foundation for the rest. Both lead papers need real new methodological and analytical work on top. I'd rather flag that now than discover it in August.

Sequencing: ship the tested, documented toolkit and the equivalence check first, as a software paper. JOSS reviews on a rolling basis, so that one can go out as soon as the port is faithful, tricky Stata tagged-missing-value handling included. Then build the standardized 2016–2024 dataset. Then write Q1 and Q4 off that foundation, using a benchmark against plain distribution-shift detectors to position Q1 rather than spending a separate paper on it. The replication is step one of this sequence either way, which is why the next concrete step at the end of this doc doesn't change.

## 6. Olivia's note

Three things in it. The first is the analysis we're replicating, predicting reported ASD diagnosis (`k2q35a_Yes`). Core of the replication, in scope. The second is the clinical-outcome pilot, with behavior therapy, ED use, and foregone care as prediction targets. That's adjacent: different targets, separate future work, and also distinct from Q4's unmet-need outcomes. The third is SOAKED, the downsampling follow-up to SOAK, and it needs clarification, because it bears directly on the fairness downsampling (the `sizes=0` downsample-to-smallest-group). There's SOAKED code and a slide deck linked in [cv-same-other-paper](https://github.com/tdhock/cv-same-other-paper) already, which is the place to start.

So there's a Phase 0 step before any fairness code gets written:

- [ ] Get the SOAKED draft from Olivia and check whether its downsampling differs from the downsample-to-smallest-group the current code does. If it does, the splitter's seeded-downsample logic and the fairness run matrix both change, and I'd rather know before building D than after.

(I'm reconstructing the three-way split from context rather than Olivia's exact words, so it's worth a quick check against the note itself.)

## 7. Decisions

Settled:

- 2024 period assignment: P5, per Olivia. This keeps the 2-year-bin pattern the prep code defines (P1=2016–17, P2=2018–19, P3=2020–21, P4=2022–23).
- Equivalence is judged on predictions, for the reasons in §4.
- One repo, `nsch-ml-py`, with the splitter as an internal module.
- Lead with Q1 paired with Q4, off the shared dataset and codebase.

Open, replication:

- [ ] **Tolerance for the held-out comparison** (gates Phase 1). My lean: mean absolute difference on the continuous predictions, threshold set once we've seen toy-data numbers, and classification agreement at 99% or better for the binary outcome (plan §8). We also need a tolerance on the per-period AUC and accuracy, and on the same/other/all error-difference p-value. Mine to propose, the co-authors to ratify.
- [ ] **svy availability.** I verify it installs at the start of Phase 1 and fall back to Polars for descriptives if it doesn't. Q1 can't start without an inference plan, so the backstop (rpy2 into R's survey, or hand-rolled Taylor linearization) needs to be named at the same time. Mine to run down, escalate if it's not there.

Open, extension (what I'd most want your read on):

- [ ] **The variance estimator for Q1.** The design-variable question is settled (§5): strata are FIPSST crossed with STRATUM, each household is its own PSU, and there are no replicate weights, so folds are stratified over households and variance comes from Taylor linearization or replicates we build ourselves. What's open is the actual estimator for a difference in cross-validated error between same/other/all under that design. This is the piece I want Ben or Toby to sanity-check before I commit to an approach.
- [ ] **Q1 tractability.** Honest uncertainty for a difference in cross-validated error under a complex design is hard. Doable on my timeline, or does it need hands-on help from Ben or Toby to do properly?
- [ ] **Raw file vintage.** Confirm the raw .dta files behind the harmonized dataset were downloaded after the April 2024 re-release of the revised 2016–2020 weights (Phase 3). Anything weighted depends on it. Mine to check.
- [ ] **Q4 outcome definitions.** How do we define unmet need and care coordination so they hold across 2016–2024, given the questionnaire changes? Needs Olivia and clinical input.
- [ ] **How hard we lean on 2024.** With the weighting revision applied retroactively across all years, the trend concern is smaller than I first thought, but the 2024 wave is still the newest and least-vetted. Feature it, or treat it as a sensitivity check?
- [ ] **Sequencing.** Software paper first (toolkit plus equivalence, to JOSS), or go straight at Q1?
- [ ] **Venues.** Software paper to JOSS; Q1 to SADM or JCGS; Q4 to Autism, JADD, or Health Services Research. The reasoning is in the memo; the call is the team's.

## 8. Out of scope

- The clinical-outcome pilot (behavior therapy, ED use, foregone care as targets). Separate future work, distinct from Q4's unmet-need outcomes.
- Causal claims. Cross-sectional, caregiver-reported data, so Q4 stays associational.
- The Medicaid-claims linkage work. A later study.
- Standalone papers for the candidate questions I'm not leading with. Q2 and Q6 fold into the software paper, Q3's data work folds into Q4, and Q5 positions Q1.
- Workstreams A, B, and D themselves. The replication consumes A's and B's output and feeds D's results, but it isn't doing their work.
- Byte-level equivalence. The bar is analytic and prediction-level agreement; identical bytes and column order are non-goals.
- Coefficient-level equivalence on the linear models, which the glmnet/sklearn differences rule out.
- The `nsch-py` data-prep port. Separate package; the replication consumes its output.
- Publishing `nsch-ml-py` to PyPI. Not needed for v1.

## What I'd do first

The replication is step one of the sequencing in §5, so the immediate move is the same either way: stand up the `nsch-ml-py` skeleton mirroring `nsch-py`, so the splitter and the data module have somewhere to live. Right before writing the one-hot encoder, I'd pull the header rows of the R `2016_2023_SOAK.csv` and `2016_2023_SOAK_categories.csv` and save them as the golden fixture. The reference rule in §4 is the highest-risk piece, and a test against that exact column set is what catches a baseline shift before it reaches the models. After that, the tolerance metric is the last thing gating Phase 1, so that's the conversation to have with Olivia and the co-authors, alongside the svy question, which the extension can't start without.

## Appendix: checking against the source

Before trusting the spec I read the original code, since vas235 has moved on and the committed scripts are the only authority left. I went through [`ml-final-prep.R`](https://github.com/vas235/ASD3-machine-learning/blob/main/ml-final-prep.R), the [core](https://github.com/vas235/ASD3-machine-learning/blob/main/ml_to_registry.R) and [fairness](https://github.com/vas235/ASD3-machine-learning/blob/main/2016_2023_fairness_SOAK/ml_to_registry.R) `ml_to_registry.R`, [`get_shap.R`](https://github.com/vas235/ASD3-machine-learning/blob/main/get_shap.R), and the [`shap_kmeans_analysis`](https://github.com/vas235/ASD3-machine-learning/blob/main/shap_kmeans_analysis) scripts ([`kmeans_shap.R`](https://github.com/vas235/ASD3-machine-learning/blob/main/shap_kmeans_analysis/kmeans_shap.R), [`best_k.R`](https://github.com/vas235/ASD3-machine-learning/blob/main/shap_kmeans_analysis/best_k.R), [`k4_cluster_final.R`](https://github.com/vas235/ASD3-machine-learning/blob/main/shap_kmeans_analysis/k4_cluster_final.R)). Most of it matched what I expected: the `k2q35a_Yes` outcome, the five learners and their tuning ranges, the inner 5-fold and outer 10-fold resamplings, the (period, outcome) stratification grouped by period, the 288-column task, the glmnet category and ablation tasks, the SHAP settings, the k=2–13 by seed=1–10 clustering grid with `nstart` 5, the fairness deletes, the metro_yn imputation, the >10%-then-complete-case missing handling, the race7/povlev4/fairness derivations, and the four period bins. Five things didn't match, and they're folded into the sections above:

1. The conditional one-hot default is the reverse of what I'd assumed. Non-reversed multi-level factors drop the first level, non-reversed binaries drop the last, and the sixteen reversed variables flip both. The reversed list itself was right. This is the one that would've quietly moved every model's baseline, so it matters most.
2. The fairness glmnet is lasso, not ridge. The core driver sets `alpha` 0; the fairness driver leaves it at the default 1. Match each separately.
3. The seeds differ by driver: core 1, fairness 42, clustering 1–10 with the final at 4. Reproducing the actual folds and labels depends on those.
4. The repo README is stale twice over. It says the clustering tests k up to 30 and reports seed 1, but the scripts say k to 13 and seed 4. The scripts win.
5. [`ml-final-prep.R`](https://github.com/vas235/ASD3-machine-learning/blob/main/ml-final-prep.R) lives in the main analysis repo rather than the [`-prep`](https://github.com/vas235/ASD3-machine-learning-prep) one. The `-prep` repo holds [`data-cleanup.R`](https://github.com/vas235/ASD3-machine-learning-prep/blob/main/data-cleanup.R) plus [`variable-config.json`](https://github.com/vas235/ASD3-machine-learning-prep/blob/main/variable-config.json), which is the harmonization layer `nsch` and `nsch-py` already replace. So the `data/` module here replicates `ml-final-prep.R` and consumes the harmonized output rather than redoing the harmonization.

Two files I haven't traced at the source yet: `rds_to_viz.R`, which produces the ROC curves, the confusion matrices at the FNR targets, and the ridge coefficient tables; and soakpy's 0.0.54 API. Neither blocks scaffolding. The reporting is easy to match once the models exist, and soakpy is reference-only.

## References

Analysis source (the work being replicated, all public):

- [`vas235/ASD3-machine-learning`](https://github.com/vas235/ASD3-machine-learning): the Monsoon analysis scripts. Key files: [`ml-final-prep.R`](https://github.com/vas235/ASD3-machine-learning/blob/main/ml-final-prep.R), [`ml_to_registry.R`](https://github.com/vas235/ASD3-machine-learning/blob/main/ml_to_registry.R) (core SOAK), [`2016_2023_fairness_SOAK/ml_to_registry.R`](https://github.com/vas235/ASD3-machine-learning/blob/main/2016_2023_fairness_SOAK/ml_to_registry.R), [`get_shap.R`](https://github.com/vas235/ASD3-machine-learning/blob/main/get_shap.R), [`shap_kmeans_analysis/`](https://github.com/vas235/ASD3-machine-learning/blob/main/shap_kmeans_analysis).
- [`vas235/ASD3-machine-learning-prep`](https://github.com/vas235/ASD3-machine-learning-prep): the harmonization layer ([`data-cleanup.R`](https://github.com/vas235/ASD3-machine-learning-prep/blob/main/data-cleanup.R), [`variable-config.json`](https://github.com/vas235/ASD3-machine-learning-prep/blob/main/variable-config.json)).
- [`vas235/MSA-imputation`](https://github.com/vas235/MSA-imputation): the metro_yn / MSA imputation method.

Package family (all public):

- [`NAU-ASD3/nsch`](https://github.com/NAU-ASD3/nsch): the R harmonization package.
- [`NAU-ASD3/nsch-py`](https://github.com/NAU-ASD3/nsch-py): the Python port; produces `get_clean_data()`, which this repo consumes.
- [`NAU-ASD3/nsch-ml-paper`](https://github.com/NAU-ASD3/nsch-ml-paper): the paper repo.
- [`NAU-ASD3/reproduce-soak-nsch`](https://github.com/NAU-ASD3/reproduce-soak-nsch): the NAU-side SOAK reproduction.

SOAK and data:

- SOAK paper: Hocking, Thibault, Bodine, Arellano, Shenkin & Lindly (2026), Same/Other/All K-Fold Cross-Validation for Estimating Similarity of Patterns in Data Subsets, Statistical Analysis and Data Mining 19(1). DOI [10.1002/sam.70055](https://doi.org/10.1002/sam.70055), preprint [arXiv:2410.08643](https://arxiv.org/abs/2410.08643).
- [`tdhock/mlr3resampling`](https://github.com/tdhock/mlr3resampling): the reference SOAK implementation. [CRAN page](https://cran.r-project.org/package=mlr3resampling) with the usage vignettes.
- [`tdhock/cv-same-other-paper`](https://github.com/tdhock/cv-same-other-paper): paper code, the NSCH-autism reproduction, and the SOAKED downsample analysis.
- NSCH_autism fixture and the other SOAK datasets: [Zenodo 10.5281/zenodo.18273949](https://doi.org/10.5281/zenodo.18273949).
- soakpy (PyPI 0.0.54): reference only, API specifics not yet confirmed.

Extension (Q1 + Q4). Full bibliography is in the Candidate Research Questions memo; the load-bearing ones:

- Design-based CV: Wieczorek, Guerin & McMahon (2022), K-fold cross-validation for complex sample surveys, Stat 11(1), e454, DOI [10.1002/sta4.454](https://doi.org/10.1002/sta4.454) (the [surveyCV package](https://github.com/ColbyStatSvyRsch/surveyCV)). Built on Lumley's survey package (2004, JSS 9(8), 1–19, DOI [10.18637/jss.v009.i08](https://doi.org/10.18637/jss.v009.i08); 2010, Complex Surveys, Wiley).
- [`svy`](https://svylab.com/docs/svy): design-based survey analysis in Python (Diallo, successor to samplics). Descriptive-only in the replication; central to the extension. Install still to be confirmed.
- Survey-aware ML: Oh et al. (2026), Survey-aware Machine Learning, a scoping review, [arXiv:2605.08963](https://arxiv.org/abs/2605.08963); MacNell et al. (2023), PLOS ONE 18(1), e0280387, on how weighting changes what a boosted-tree model learns.
- Q4 anchor: Lindly, Chavez & Zuckerman (2016), unmet health services needs among US children with developmental disabilities, J. Dev. Behav. Pediatr. 37(9), 712–723.

NSCH design and weighting documentation (the §5 design-variable facts and the Phase 3 weighting caution come from these):

- [NSCH Analytic Guide](https://www2.census.gov/programs-surveys/nsch/technical-documentation/methodology/NSCH-Analytic-Guide.pdf) (Census): variance design variables (strata = FIPSST × STRATUM, PSU = HHID), multi-year stratum handling.
- [NSCH Weighting Revisions technical document](https://www2.census.gov/programs-surveys/nsch/technical-documentation/NSCH_Weighting_Revisions.pdf) (Census): the race/ethnicity imputation and weighting revision introduced with the 2022 NSCH and applied retroactively to 2016–2021.
- [2024 NSCH FAQs](https://www2.census.gov/programs-surveys/nsch/technical-documentation/methodology/2024-NSCH-FAQs.pdf) (Census): the same design variables for the newest wave.
