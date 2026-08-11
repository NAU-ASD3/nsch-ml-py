# Why the Python results differ from the R results

Chris Reger, 10 August 2026. Working note for the ASD3 methods paper.

## Short version

Everything upstream of model fitting matches exactly: rows, folds, train and test membership on all 60 splits, downsample counts on all 40 downsampled splits, all checked index for index against the archived R fixture.

Comparing the fitted models turned out to be harder than it looks, for a reason that took us most of a week to see. The R runs we had did not all partition the data the same way. Comparing two runs fold by fold assumes fold 7 holds the same children on both sides, and across some pairs of runs it does not. Where that assumption fails, the comparison measures the partitioning rather than the models.

Once runs are compared like for like, everything agrees closely. On mean AUC within each of the six cells, a quantity that does not depend on how folds are numbered, all eleven runs we have sit within 0.0005 of each other, and that includes the results the SOAK paper was published from. Against an R run that shares our fold assignment, our port sits 0.0021 away fold by fold.

Against margins fixed in writing before the comparison was run, the port passes 58 of 60 splits. The two failures are one fitted model, and they come from a rank criterion poorly suited to an outcome with a 3% base rate.

## What is identical

|                                                    | Checked against                          |
| -------------------------------------------------- | ---------------------------------------- |
| 46,010 rows, 364 features                          | `data_Classif/NSCH_autism.csv`           |
| Fold assignment                                    | `nsch_autism_folds.csv`, used directly   |
| Train and test rows, 60 full splits                | `nsch_autism_iterations_long.csv`, exact |
| Downsample sizes and per-stratum counts, 40 splits | same, exact                              |

The fold fixture was drawn by `mlr3resampling` 2026.5.19 under seed 1, recorded in `nsch_autism_fixture_provenance.csv`. Our splitter reads it directly, so our folds are that draw. Which R runs share it matters, and the next section is about that.

Row membership of downsampled train sets does not match and never could, since R's `sample()` and NumPy's generator are different machines. The rule matches; the draw does not. Only full splits feed the model comparison.

## The R runs do not all partition the data the same way

We have ten R runs of this analysis plus the published results. Comparing them fold by fold, they fall into three groups: five February and March runs, four August runs, and batchtools on its own. Within a group the mean absolute difference in per-fold AUC is about 0.0005. Between groups it is about 0.0073.

We first read that as a difference in the analysis, and said so in earlier drafts of this note. It is not.

Two things pointed the other way. Mean AUC within each cell, which does not depend on how folds are numbered, shows no such separation: across all 55 pairs of runs the largest gap is 0.00051 and the median is 0.00027. And optimal matching, which finds the pairing of folds that minimises the distance, closes 64 to 66 percent of the gap between groups while closing only 5 to 23 percent within a group.

That second number is the one that settles it. Matching always helps a little by chance, and pairs sharing a fold assignment show how much: 5 to 23 percent. Across the boundary it helps three times as much, with no overlap between the two ranges over 44 pairs. The folds are the same children under different numbering.

`NSCH.R` draws the folds at line 50, `set.seed(1)` followed by `SOAK$instantiate(task.obj)`, so the draw belongs to `mlr3resampling`. Something about it changed between February and August. We have not established what, and no February artifact records its fold assignment, so this remains an inference from the AUC values rather than a direct comparison of the assignments themselves.

Batchtools sits apart from its own February siblings by the same measure, dropping 69 to 71 percent under matching. It has a third fold assignment, which is what dispatching jobs to separate processes would do to a draw that depends on the order the random stream is consumed.

### What does hold across all of them

| Comparison                        | Mean absolute difference in cell means |
| --------------------------------- | -------------------------------------- |
| Smallest pair                     | 0.000000                               |
| Median across all 55 pairs        | 0.000273                               |
| Largest pair                      | 0.000513                               |
| Published results to nearest run  | 0.000188                               |
| Published results to furthest run | 0.000419                               |

`data_Classif_batchmark_registry.csv` in the cv-same-other-paper repository holds the results the SOAK paper was written from. Its NSCH_autism rows give 0.967000 training on all and 0.965770 training on same for the 2020 subset, against the 0.9670 and 0.9658 printed in Section 4.3. It used `same_other_cv`, the resampling class since removed, so its folds correspond to nothing we have and only its cell means are comparable. On those it sits in the middle of the pack.

So the analysis is stable at the level the paper reports it. What moves between runs is which children land in which fold, and that moves individual fits without moving what gets published.

### The inner seed

Three August runs differ only in the `cv.glmnet` seed. They agree to 0.00028 to 0.00059 on per-fold AUC, comparable to two runs on different machines. `NSCH_unseeded` is bit-identical to `NSCH_seed1` on all 60 splits, because the learner defaults to seed 1, so February's runs were effectively seeded all along.

## Which reference to use

Everything below compares against `NSCH_seed1`. That run shares our fold assignment, verified directly rather than inferred: joining its held-out predictions to ours on split and row identifier matched all 138,030 rows, with the outcome agreeing on every one.

Earlier drafts compared against `NSCH_proj`, from February. Those figures are kept in `analyses/glmnet_replication_grid60.csv` but should not be read as a measure of anything. `NSCH_proj` uses a different fold assignment, so pairing our fold 7 against its fold 7 compared different children, and the 0.0071 that comparison produced is a partition mismatch rather than an implementation difference.

## What differs, and why

**Penalty selection, which is most of it.** `cv.glmnet` derives a lambda path from the data, roughly 100 values, runs an internal 10-fold cross-validation, scores by binomial deviance, and predicts at `lambda.1se`. We run 5-fold cross-validation over a fixed 60-point grid of `C`, score by log loss, and apply the same one-standard-error rule. Different candidates, different inner folds, different scoring scale. These are two models chosen by similar but distinct procedures, not one model computed twice.

**Optimizer.** glmnet uses cyclical coordinate descent. We use `liblinear`, which is also coordinate descent, having previously used `saga`. The two scikit-learn solvers select penalties within one or two grid steps of each other and agree on AUC to 0.0006, so the change was for speed rather than fidelity: 72 minutes for 60 splits against 11.5 hours.

**The intercept.** `liblinear` penalizes the intercept; glmnet and `saga` do not. Holding the penalty fixed and varying `intercept_scaling`, the intercept sits 0.36 from `saga`'s at the default and converges to within 0.002 by a scaling of 100, while held-out probabilities move by 0.0017. Running all 60 splits at 100 tightened fold-level agreement by about a quarter.

**Parameterization.** glmnet minimizes `(1/n) * deviance + lambda * penalty`; scikit-learn minimizes `C * loss + penalty`. So lambda is roughly `1/(nC)`, and the grids align only approximately.

Two candidate explanations were tested and ruled out. Penalty family is not the cause: ridge, lasso, and a 60-point ridge grid put the 2019 All-minus-Same estimate within 0.0006 of each other. Grid coarseness is not the cause either: going from 12 penalty values to 60 changed the selected penalty by about 1.35x and moved nothing of substance.

## Size of the difference, split by split

Paired t on 9 degrees of freedom against `NSCH_seed1`, using our ridge fits:

| Test subset | Train source | Mean AUC difference | 95% interval             | p         |
| ----------- | ------------ | ------------------- | ------------------------ | --------- |
| 2019        | Same         | −0.00024            | (−0.00305, +0.00258)     | 0.854     |
| 2019        | Other        | −0.00108            | (−0.00374, +0.00158)     | 0.383     |
| 2019        | All          | −0.00085            | (−0.00282, +0.00111)     | 0.352     |
| 2020        | Same         | +0.00071            | (−0.00089, +0.00232)     | 0.340     |
| 2020        | Other        | **+0.00205**        | **(+0.00056, +0.00353)** | **0.012** |
| 2020        | All          | +0.00055            | (−0.00057, +0.00167)     | 0.297     |

Five of six show no detectable difference. The sixth, 2020 Other, has an interval excluding zero at a raw p of 0.012. Six cells were tested, so under either Bonferroni or Benjamini-Hochberg that becomes 0.074: suggestive, not established. A document that argues for multiplicity correction does not get to skip it for its own findings.

Worth watching rather than concluding. If it is real it is about two thousandths of AUC, which changes no one's conclusions, but it would be the first sign of a systematic difference rather than scatter.

## The criterion problem

Toby's two-sided paired t on 9 degrees of freedom, per test subset, on both implementations:

| Contrast     | Subset | R estimate | R p    | Python estimate | Python p |
| ------------ | ------ | ---------- | ------ | --------------- | -------- |
| All − Same   | 2019   | +0.00301   | 0.0015 | +0.00239        | 0.0326   |
| All − Same   | 2020   | +0.00163   | 0.0006 | +0.00146        | 0.0037   |
| Other − Same | 2019   | +0.00164   | 0.0903 | +0.00079        | 0.5531   |
| Other − Same | 2020   | −0.00206   | 0.0139 | −0.00073        | 0.1654   |

Three of four verdicts match. Correcting for multiplicity, which we should, since this is four tests and the full analysis will run a couple of hundred:

| Adjustment         | Verdicts agreeing | Which one disagrees  |
| ------------------ | ----------------- | -------------------- |
| None               | 3 of 4            | Other − Same, 2020   |
| Bonferroni         | 3 of 4            | **All − Same, 2019** |
| Benjamini-Hochberg | 2 of 4            | both of the above    |

The count barely moves. The comparison does. Which result disagrees depends on a correction chosen after seeing the data.

**And R fails this criterion against itself.** Across the ten R runs, two of four contrasts get different verdicts depending on which run you use. The February `mpi` run calls 2019 Other-minus-Same significant at 0.0196 while four sibling runs sharing its fold assignment call it null between 0.059 and 0.107. One August run landed at 0.0502, two ten-thousandths from flipping.

Verdict agreement is not a standard the reference analysis meets against itself. It cannot reasonably be required of a port.

### The two scales

|                                             | Value   |
| ------------------------------------------- | ------- |
| Typical SOAK contrast under test            | 0.00163 |
| Typical R-versus-Python interval half-width | 0.00179 |
| Ratio                                       | 1.1x    |

Both are medians over both implementations and all six cells, against `NSCH_seed1`. The gap between the two implementations is about the same size as the effect being tested, which is not a comfortable place to be running significance tests from even though it is much better than we thought a week ago.

## Where this leaves the criterion question

Report the paired t-test p-values, since they answer the scientific question and they are the method the SOAK paper prescribes. The open question was what standard the port should be held to, and this work tried three candidates on fold-level AUC:

| Candidate                                       | Result against `NSCH_seed1`    |
| ----------------------------------------------- | ------------------------------ |
| Verdicts agree, uncorrected                     | 3 of 4                         |
| Verdicts agree, Bonferroni                      | 3 of 4, a different comparison |
| Verdicts agree, Benjamini-Hochberg              | 2 of 4                         |
| Each estimate falls inside the other's interval | 3 of 4                         |
| The two intervals overlap                       | 4 of 4                         |

None is fit to be the standard, and each fails differently. Verdict agreement is not met by the R runs against each other, and which comparison fails moves with the multiplicity adjustment. Estimate containment gets harder to satisfy as intervals tighten, so a better-measured comparison can fail where a noisier one passed. Interval overlap has the opposite defect: it is the weakest of the three, known to have poor power as an equivalence check (Schenker and Gentleman 2001), and it is not reassuring that the most lenient candidate is the one that passes.

There is a structural problem underneath all three. Every candidate was evaluated after its results were visible, in a document arguing that conclusions should not depend on choices made after seeing the data. And all three operate on fold-level AUC, ten numbers per cell summarising 1,819 predictions each, with the added caveat that paired t-tests on overlapping cross-validation folds understate variance (Dietterich 1998; Nadeau and Bengio 2003; Bengio and Grandvalet 2004), so the p-values above are optimistic on both sides.

The resolution was to change the quantity rather than keep auditioning tests: compare the predicted probabilities themselves, against a margin fixed in writing before the comparison was computed. That margin is in `docs/equivalence-margin.md` and the result is below.

**And record the fold assignment.** A margin means nothing without knowing that both sides partitioned the data the same way. That is the lesson replacing the one this document used to draw about pinning the package build.

## The comparison against the pre-registered margins

Run on 7 August against `NSCH_seed1`, using our lasso fits with `liblinear` at `intercept_scaling=100`. The margins were committed the day before the comparison script was written.

R's `cv_glmnet` predicts at `lambda.1se`, so that is the gated column. `lambda.min` is reported beside it.

| Quantity                                 | Observed | Margin        |          |
| ---------------------------------------- | -------- | ------------- | -------- |
| Spearman of held-out scores, worst split | 0.94057  | at least 0.95 | **fail** |
| Probability MAD, worst split             | 0.006554 | at most 0.01  | pass     |
| Fold-level AUC, mean absolute            | 0.001271 | at most 0.002 | pass     |

Fifty-eight of sixty splits clear every gated check. Two do not.

### The two failures are one fitted model

They are 2019 "same" fold 7 at 0.94057 and 2020 "other" fold 7 at 0.94106. Those two cells train on identical rows, and their R coefficients differ by exactly zero, so this is one fit scored on two test subsets rather than two independent failures.

Nothing about the fit stands out. It holds out 1,821 children, 55 of whom have the outcome, a rate of 0.0302 that matches folds 2, 6, 8 and 9 exactly. It keeps 19 features, as do folds 3, 5, 6 and 10, three of which pass comfortably.

### What Spearman is measuring here

The children the two implementations rank most differently are all true negatives at probabilities below 0.007:

| row_id | outcome | R       | Python  | rank displacement |
| ------ | ------- | ------- | ------- | ----------------- |
| 1587   | 0       | 0.00345 | 0.00087 | 972               |
| 1252   | 0       | 0.00216 | 0.00236 | 732               |
| 13409  | 0       | 0.00216 | 0.00235 | 728               |
| 1439   | 0       | 0.00216 | 0.00234 | 726               |

The three consecutive rows are the giveaway. R returns exactly 0.00216 for all of them and Python returns three slightly different values, so R has produced a tie and Python has broken it. Spearman gives tied observations their average rank, which means any ordering imposed on them counts as a large displacement. On this split 828 of 1,821 children move more than 100 ranks.

At a 3% base rate, 1,766 of those children are ones the model is confident about. Ranking them against each other says nothing about whether the two implementations agree on who is at risk, and that is where nearly all of Spearman's power goes. The probability MAD for the same split is 0.0058, well inside its margin.

### What this means

The margin stands. It was fixed before the comparison existed, and moving it now to accommodate a result we have already seen would forfeit the whole point of fixing it.

The reading we can defend is that the port reproduces the R analysis on 58 of 60 splits, and that the criterion which failed was a poor choice for a rare outcome. That is a fault in how we designed the check rather than evidence about the port, and the other two gated quantities pass on all 60 splits including this one.

Three changes for the next pre-registration, to be fixed in advance again rather than applied backwards to this one:

- compute rank agreement over the children the model ranks highest, not over the full held-out set, because the ordering of confident negatives is uninformative and swamps the statistic;
- make probability MAD primary rather than secondary, since it is the quantity that answers whether two implementations produce the same predictions;
- decide how to handle ties in the reference, because a coarser probability resolution on one side penalises the other for having a finer one.

`lambda.min` points the same way from a different angle. Its rank agreement is much worse, averaging about 0.93 with a worst split of 0.866, while its probability MAD stays near 0.008. Gating on `lambda.1se`, which is what `cv_glmnet` predicts at, was the right call and now has evidence behind it.

### One offset we cannot yet explain

Python's fold-level AUC runs about 0.0012 above R's at `lambda.1se`, consistently, with a paired t of +5.56 across the 20 fold clusters. That is half the margin, so it passes, but it is systematic rather than noise.

It is not a bias in the probabilities themselves. The mean signed difference is −0.0001, near enough to zero. Neither implementation produces uniformly higher predictions; they order children slightly differently, and the difference happens to favour ours.

Penalty selection is the obvious suspect. `cv.glmnet` searches a lambda path derived from the data while we search a fixed 60-point grid with different inner folds and a different scoring loss. A grid that tends to land on a marginally better-performing penalty would produce this pattern. Untested so far.

## What we do not know yet

What changed the fold draw between February and August. The draw belongs to `mlr3resampling`, so a version change there is the obvious candidate, but we have not tested it and February's fold assignment is not recorded anywhere we can read.

Why 2020 Other specifically. It is the one cell whose interval excludes zero, though not after multiplicity correction, and nothing about the penalty-selection story predicts a single cell rather than a general shift.

Whether any of this holds for other learners. Everything here is `cv_glmnet`. Boosted trees randomize differently and may behave differently.

## What we got wrong

Earlier drafts of this note led with a claim that the `mlr3learners` build shifts the R results by 0.0073 mean absolute AUC, larger than the effects the study reports. That was the most striking thing we had and it was an artifact of comparing runs that partitioned the data differently. The correction is in the second section.

Three smaller corrections, in the order they were made:

- The reference was switched from `NSCH_proj` to `NSCH_seed1`. That change was right and the reason given for it was wrong: it fixed a fold mismatch, not a build mismatch.
- The implementation half-width was originally the upper interval bound rather than `(hi - lo) / 2`. Those coincide only when the mean sits at zero, and the error overstated the ratio.
- Two scripts computed "typical contrast" over different sets and printed different ratios under the same label. Both now median over both implementations.

Printouts kept from earlier in the work will not match this document.

## Reproducing

    NSCH_SOAK_REFERENCE=$REPRO/results/seed-variation/NSCH_seed1.csv \
      uv run python analyses/run_glmnet_replication.py --full-only --lasso \
      --l1-solver liblinear --intercept-scaling 100 \
      --out analyses/glmnet_replication_lasso_seed1_is100.csv \
      --save-predictions $REPRO/results/predictions/python_lasso_seed1_is100_predictions.csv

    uv run python analyses/prediction_equivalence.py \
      --r-predictions      $REPRO/results/predictions/NSCH_seed1_predictions_repaired.csv \
      --python-predictions $REPRO/results/predictions/python_lasso_seed1_is100_predictions.csv \
      --python-auc         analyses/glmnet_replication_lasso_seed1_is100.csv

    uv run python analyses/cell_mean_distances.py --reproduce-dir $REPRO \
      --registry $PAPER/data_Classif_batchmark_registry.csv

    uv run python analyses/diagnose_fold_relabelling.py --reproduce-dir $REPRO

`$REPRO` is the reproduce-soak-nsch checkout and `$PAPER` the cv-same-other-paper one. The R runs come from `NSCH_seed_variation.R`, `NSCH_unseeded_check.R` and `NSCH_save_predictions.R`, about 25 minutes each on 8 cores.

The 60-split Python lasso run takes 72 minutes with `liblinear`. The same run with `saga` took eleven and a half hours, which is worth recording: glmnet's coordinate descent handles this problem in minutes where scikit-learn's stochastic solver needs most of a day.
