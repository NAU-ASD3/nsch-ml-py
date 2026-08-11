# Does the Python port reproduce the R analysis?

Chris Reger, 10 August 2026. Working note for the ADSI methods paper.

## What this document is

We are rewriting an existing statistical analysis from R into Python. This
note reports how closely the Python version matches the R version, and what
that means for the work built on top of it.

It assumes no familiarity with the original study or with the packages
involved. The next two sections supply the background.

## Background

### The study

The National Survey of Children's Health asks families across the United
States about their children. An earlier analysis on this project, built by a
former analyst on the team, used it to predict which children have been
identified as autistic, from about 364 survey-derived features covering
diagnoses, services, family circumstances, and demographics.

Two details shape everything below. The outcome is rare, present in about 3%
of children, and the survey runs annually, so the data arrives in yearly
batches that may or may not be safe to pool.

### SOAK, the cross-validation design

That second point is what SOAK addresses. It is a cross-validation scheme
from Hocking et al. (2026), developed by a collaborator on this project, and
it asks a specific question: when data comes in groups, does training on all
the groups help or hurt compared with training on just one?

The setup, using the two survey years in our validation dataset:

- Hold out one year as the test set, say 2019.
- Train three models. **Same** trains on other 2019 children only. **Other**
  trains on 2020 children only. **All** trains on both.
- Score all three on the same held-out 2019 children and compare.

If All beats Same, pooling years helps. If Other does much worse than Same,
the years differ in ways that matter. Repeat with 2020 as the test set.

Cross-validation means this is done ten times over. The data is divided into
ten folds; each fold takes a turn as the test set while the rest supply
training data. So the full experiment is 2 test years times 3 training
strategies times 10 folds, which is **60 splits**. Each split trains one
model and scores it on roughly 1,800 to 2,800 held-out children.

The paper reports a contrast: the average difference between two training
strategies across the ten folds, with a paired t-test. Those contrasts are
small, around 0.002 in AUC, which becomes important later.

### The model

The original analysis fits penalized logistic regression using R's
`cv.glmnet`. Penalized means coefficients are shrunk toward zero to avoid
overfitting 364 features. How much shrinkage is controlled by a parameter,
lambda, and `cv.glmnet` picks it by running its own cross-validation inside
each training set.

Two flavours of penalty exist. **Lasso** can shrink coefficients exactly to
zero, dropping features entirely. **Ridge** shrinks everything but drops
nothing. `cv.glmnet` defaults to lasso, and we confirmed that from the
original run's saved coefficients: 65 of 364 features have nonzero weights,
and 255 are never selected in any split.

`cv.glmnet` also offers two rules for choosing lambda. `lambda.min` picks
whatever scored best; `lambda.1se` picks a stronger penalty one standard
error away, which is more conservative. The R code predicts at `lambda.1se`,
so that is the setting we compare against.

## What "reproduce" means here, and why it matters

The project needs to extend this analysis to 2024 data and to three new
outcomes. That work happens in Python. If the Python code does not reproduce
the R results on the existing data, then when the new numbers differ from the
old ones, nobody can say whether the data changed or the code did.

So the port has to be checked before it is used, and checked against a
standard set in advance. Otherwise the temptation is to keep trying measures
until one passes, which proves nothing.

Everything below concerns `cv_glmnet`, the linear model. The analysis also
used four other learners, none of which has been ported or checked.

## Summary of findings

Everything up to the point of fitting a model matches exactly: the same rows,
the same features, the same children in the same folds.

The fitted models differ by a level shift of about 0.001 in AUC, consistent
across all six combinations of test year and training strategy. Because a
SOAK contrast is a difference between two combinations, that shift cancels
out and the reported science is unaffected.

Against a standard fixed in writing before the comparison ran, the port
passes 58 of 60 splits. The two failures turn out to be a single fitted model
counted twice. They come from a poorly chosen measure, not from anything the
port does wrong.

Getting here took longer than it should have, for a reason worth stating
plainly: the reference is not one thing. Several R runs of the same analysis
exist and they do not all divide the data into folds the same way. Comparing
two runs fold by fold only means something when they do.

## What matches exactly

|                                                    | Checked against                          |
| -------------------------------------------------- | ---------------------------------------- |
| 46,010 rows, 364 features                          | `data_Classif/NSCH_autism.csv`           |
| Which children are in which fold                   | `nsch_autism_folds.csv`, used directly   |
| Train and test membership, all 60 splits           | `nsch_autism_iterations_long.csv`, exact |
| Downsample sizes and per-stratum counts, 40 splits | same, exact                              |

We do not re-derive the fold assignment in Python. We read the one the R
analysis produced, so this part is exact by construction. The table records that we checked it
anyway.

One thing cannot match. SOAK also runs a downsampled variant, where the
larger training sets are randomly trimmed so all three strategies see the
same number of children. R's random number generator and NumPy's are
different machines, so the rule for how many to keep matches exactly while
the particular children drawn do not. Those splits are excluded from the
model comparison for that reason.

## The reference is not one thing

Ten R runs of this analysis are available, plus the results the paper was
published from. Comparing them fold by fold, they fall into three groups:
five runs from February and March, four from August, and one from a
particular cluster-computing backend. Runs inside a group agree to about
0.0005 mean absolute difference in AUC. Runs in different groups differ by
about 0.0073.

An earlier draft of this note read that as evidence that a package update had
changed the analysis, and led with it. That was wrong, and the correction is
worth following because the same trap is easy to fall into again.

**First clue.** Average the AUC within each of the six combinations of test
year and training strategy. That average does not depend on how the folds
happen to be numbered. On that quantity nothing separates the groups: across
55 pairs of runs the largest gap is 0.00051, median 0.00027.

**Second clue.** Take two runs and, instead of pairing fold 1 with fold 1,
find the pairing of folds that makes them agree as closely as possible. If
two runs really use the same folds, matching them optimally can only help a
little, by chance. If one run's folds are a shuffled version of the other's,
matching recovers the correspondence and helps a lot.

Within a group, optimal matching closes 5 to 23 percent of the distance.
Across groups it closes 64 to 66 percent. No overlap across 44 pairs.

So the groups use the same children in differently numbered folds. Pairing
fold 7 of one run against fold 7 of another compared different children, and
the 0.0073 measured that mismatch instead of any difference in the analysis.

The fold assignment is drawn by the `mlr3resampling` package, at a single
line in the R driver that sets a random seed and then instantiates the
resampling. Something changed that draw between February and August. We have
not established what, and no February run recorded its fold assignment, so
this remains an inference from the AUC values, not a direct check.

The cluster-computing run sits apart from its own February siblings by the
same measure. It has a third fold assignment, which is what running jobs in
separate parallel processes does to a draw that depends on the order the
random stream is consumed.

### What holds across every run

| Comparison                        | Mean absolute difference in cell averages |
| --------------------------------- | ----------------------------------------- |
| Smallest pair                     | 0.000000                                  |
| Median across all 55 pairs        | 0.000273                                  |
| Largest pair                      | 0.000513                                  |
| Published results to nearest run  | 0.000188                                  |
| Published results to furthest run | 0.000419                                  |

The published results are in the SOAK paper's own repository. For the 2020
test year they give 0.967000 training on All and 0.965770 training on Same,
against the 0.9670 and 0.9658 printed in the paper. They used an older
resampling class, so their folds correspond to nothing we have and only the
averages compare. On those they sit in the middle of the pack.

The analysis is stable at the level the paper reports it. What moves between
runs is which children land in which fold.

### The seed inside the model

Three of the August runs differ only in the seed controlling `cv.glmnet`'s
internal cross-validation. They agree to 0.00028 to 0.00059, about what two
runs on different machines give. One further run left that seed unset and
came out bit-identical to the run with it set to 1, because 1 is the default.

## What we compare against

`NSCH_seed1`, an August run sharing our fold assignment. We verified that
directly: joining its per-child predictions to ours by split and row
identifier matched all 138,030 rows, with the recorded outcome agreeing on
every one.

Every number below comes from a single Python run, our lasso fits saved in
`analyses/glmnet_replication_lasso_seed1_is100.csv`. An earlier draft mixed
figures from two different runs in the same tables.

## Where the two implementations differ

**Choosing the penalty, which is most of it.** `cv.glmnet` builds its own
list of candidate lambdas from the data, roughly 100 of them, and evaluates
them with 10-fold cross-validation scored by binomial deviance. We evaluate a
fixed 60-point grid with 5-fold cross-validation scored by log loss. Similar
procedures, different candidates, different inner folds, different scoring
scale. The two sides are not the same model computed twice.

**Penalty family, which we had wrong for a while.** Our early runs used
ridge. `cv.glmnet` defaults to lasso. Switching roughly halved the measured
gap, from 1.1 times the size of the contrasts down to 0.5.

**Optimizer.** R's glmnet uses an algorithm called coordinate descent. We
originally used a stochastic solver called `saga` and switched to
`liblinear`, which is also coordinate descent. The two Python solvers pick
almost the same penalty and agree on AUC to 0.0006, so the switch bought
speed: 72 minutes for 60 splits against 11.5 hours.

**The intercept.** `liblinear` shrinks the model's intercept along with
everything else; glmnet does not. A setting called `intercept_scaling`
controls how much. Raising it to 100 brings our intercept to within 0.002 of
the unshrunk value and moves predicted probabilities by 0.0017. All 60 splits
use that setting.

**Parameterization.** The two libraries express penalty strength inversely to
each other, so our grid aligns with glmnet's path only approximately.

### Three conventions about which class is "positive"

None of these is visible in an AUC, and any one of them missed would flip
every predicted probability while leaving the AUC untouched.

The outcome column in the data holds the strings "Yes" and "No", not 1 and 0.
The mlr3 framework treated "No" as the positive class, so R's saved
probabilities are one minus the probability of the outcome, and its
coefficients point the same way. scikit-learn, handed string labels, treats
the alphabetically last as positive, which is "Yes".

They happen to compose correctly once the probability is inverted. That is
luck, and the code documents it so nobody has to rediscover it.

## The result

The standard is in `docs/equivalence-margin.md`, committed the day before the
comparison script was written. Three quantities, each with a threshold fixed
in advance:

| Quantity                                 | Observed | Threshold     |          |
| ---------------------------------------- | -------- | ------------- | -------- |
| Spearman of held-out scores, worst split | 0.94057  | at least 0.95 | **fail** |
| Probability MAD, worst split             | 0.006554 | at most 0.01  | pass     |
| Fold-level AUC, mean absolute            | 0.001271 | at most 0.002 | pass     |

Spearman correlation asks whether the two implementations rank the held-out
children in the same order. MAD is the mean absolute difference between the
two predicted probabilities for the same child.

Fifty-eight of sixty splits clear every check. Across all 60, the Spearman
correlation averages 0.980 with a median of 0.984.

### The two failures are one fitted model

They are the 2019 Same strategy at fold 7, and the 2020 Other strategy at
fold 7. Those two cells train on exactly the same children, so they produce
the same model, scored on two different test years. Their R coefficients
differ by exactly zero.

Nothing about that model stands out. It holds out 1,821 children, 55 with the
outcome, a rate matching four other folds exactly. It keeps 19 features, as
do four other folds, three of which pass comfortably.

Here are the children the two implementations rank most differently:

| row_id | outcome | R       | Python  | rank displacement |
| ------ | ------- | ------- | ------- | ----------------- |
| 1587   | 0       | 0.00345 | 0.00087 | 972               |
| 1252   | 0       | 0.00216 | 0.00236 | 732               |
| 13409  | 0       | 0.00216 | 0.00235 | 728               |
| 1439   | 0       | 0.00216 | 0.00234 | 726               |

All are children without the outcome, given probabilities under 0.007 by both
sides. The three consecutive rows give the problem away. R assigns exactly
0.00216 to all three, a tie; Python assigns three slightly different values
and breaks it. Spearman gives tied observations their average rank, so any
ordering imposed on a tie registers as a large displacement. On this split
828 of 1,821 children move more than 100 ranks.

At a 3% base rate, 1,766 of those 1,821 children are ones the model is
confident about. Their relative ordering says nothing about whether the two
implementations agree on who is at risk, and that is where nearly all of
Spearman's power goes. The probability MAD for this same split is 0.0058,
well inside its threshold.

The threshold stands. It was fixed before the comparison existed, and
adjusting it now to accommodate a result we have already seen would forfeit
the point of fixing it. The honest reading is that the port reproduces the
analysis on 58 of 60 splits, and that the measure which failed suited
a rare outcome badly.

Three changes for next time, again fixed in advance, not applied
backwards here:

- measure rank agreement only among the children the model ranks highest,
  since the ordering of confident negatives is uninformative and swamps the
  statistic;
- make probability MAD the primary check;
- decide how to handle ties in the reference, since a coarser probability
  resolution on one side penalises the other for having a finer one.

Choosing `lambda.1se` was right. Its worst probability MAD is 0.0066, inside
the threshold; `lambda.min`'s worst is 0.0131, which would fail it.

### The offset, and why it matters less than it looks

Our AUC runs above R's in all six cells, five of them significantly:

| Test year | Training strategy | Mean difference | 95% interval         | p      |
| --------- | ----------------- | --------------- | -------------------- | ------ |
| 2019      | Same              | +0.00099        | (+0.00031, +0.00167) | 0.0096 |
| 2019      | Other             | +0.00124        | (+0.00011, +0.00236) | 0.0346 |
| 2019      | All               | +0.00067        | (−0.00006, +0.00141) | 0.0681 |
| 2020      | Same              | +0.00115        | (+0.00012, +0.00217) | 0.0327 |
| 2020      | Other             | +0.00181        | (+0.00095, +0.00268) | 0.0010 |
| 2020      | All               | +0.00112        | (+0.00051, +0.00173) | 0.0024 |

This is a level shift of about +0.001, varying across cells by roughly
±0.0003. A SOAK contrast is a difference between two cells, so the shift
cancels and only the variation survives.

The arithmetic confirms it. For the 2019 All-minus-Same contrast, the two
cell offsets differ by 0.00067 − 0.00099 = −0.00032, and the R-versus-Python
gap on that contrast is −0.00031.

It is not a bias in the probabilities themselves: the mean signed difference
across all 138,030 predictions is −0.0001. The two implementations order
children slightly differently, and the difference happens to favour ours.

Penalty selection is the obvious suspect, since that is where the procedures
differ most. Untested.

## Two ways of scoring, and why we prefer one

The SOAK paper's own figures compute their p-values on percent error, the
share of children the model classifies wrongly. Our comparison used AUC, the
probability that a randomly chosen child with the outcome is ranked above a
randomly chosen child without it. When the team asked us to use the paper's
p-values, percent error is what they meant.

Both come out of the same results file, so this cost no new computation.
Higher AUC is better; higher error is worse, so the signs run opposite ways.

The contrasts on AUC:

| Contrast     | Test year | R estimate | R p    | Python estimate | Python p |
| ------------ | --------- | ---------- | ------ | --------------- | -------- |
| All − Same   | 2019      | +0.00301   | 0.0015 | +0.00270        | 0.0051   |
| All − Same   | 2020      | +0.00163   | 0.0006 | +0.00160        | 0.0038   |
| Other − Same | 2019      | +0.00164   | 0.0903 | +0.00189        | 0.0594   |
| Other − Same | 2020      | −0.00206   | 0.0139 | −0.00140        | 0.0931   |

The same contrasts on percent error:

| Contrast     | Test year | R estimate | R p    | Python estimate | Python p |
| ------------ | --------- | ---------- | ------ | --------------- | -------- |
| All − Same   | 2019      | −0.04393   | 0.3662 | −0.01649        | 0.6782   |
| All − Same   | 2020      | −0.01078   | 0.8386 | −0.00719        | 0.8751   |
| Other − Same | 2019      | −0.03295   | 0.3703 | −0.01099        | 0.7752   |
| Other − Same | 2020      | +0.14385   | 0.0154 | +0.16183        | 0.0019   |

On percent error, R and Python reach the same verdict on all four contrasts.
On AUC they agree on three. That looks like a point in favour of percent
error until you notice that three of the four contrasts are null on both
sides, with p-values from 0.37 to 0.88. The agreement is agreement about
nothing.

Look at what the metric misses. On AUC, the 2019 All-minus-Same contrast is
R's strongest result at p = 0.0015: pooling years helps. On percent error the
same contrast is p = 0.3662. The paper's central finding is invisible.

The reason is the base rate. With the outcome present in about 3% of
children, a model that answers "no" for everyone is already 97% accurate.
Observed accuracy across all 60 splits runs from 0.973 to 0.980, so percent
error has almost no room to move, and most of the room it has is the base
rate rather than the model.

The two scales side by side:

|                                     | AUC     | Percent error |
| ----------------------------------- | ------- | ------------- |
| Typical contrast being tested       | 0.00176 | 0.02472       |
| Typical R-versus-Python uncertainty | 0.00080 | 0.08673       |
| Ratio                               | 0.5x    | 3.5x          |

On AUC the difference between implementations is half the size of the effect
being tested. On percent error it is three and a half times larger, so the
comparison is drowned out.

So AUC is our primary measure and percent error is reported because it is the
paper's. Notably the paper itself reports both accuracy and AUC for this
particular dataset, which suggests the same reservation. Worth putting to the
SOAK author directly: would he use error on an outcome this rare?

## How good is the port, and what can it be used for

**Exact** in everything before the model is fitted, verified row by row.

**Close** on results. Predicted probabilities agree to 0.0066 on the worst
split against a 0.01 threshold. Fold-level AUC agrees to 0.0013 against
0.002. Both implementations reach the same conclusion on the contrast the
paper reports.

**Shifted** by about +0.001 in AUC, consistently, varying by ±0.0003. The
shift cancels in the contrasts.

**Limited** in one place: rank agreement over the full held-out set fell
below its threshold on one of thirty distinct fitted models, for the
tie-breaking reason above.

### What follows

**For the extension, usable.** Adding 2024 and three new outcomes produces
results with no prior counterpart, so the question is whether the method is
implemented correctly, not whether it matches a previous number. The evidence
there is strong: exact splits, verified conventions, and agreement with the
published results to 0.0002.

**For reproducing the published figures, usable with the shift disclosed.**
It sits in the fourth decimal of quantities the paper reports to four
decimals.

**For any claim resting on a contrast near p = 0.05, not usable.** Neither is
the R analysis. The 2020 Other-minus-Same contrast is significant for R at
0.0139 and null for us at 0.0931, from estimates 0.0007 apart. That is what
ten folds and an effect of 0.0018 buy you.

### What does not transfer

All of this concerns one model. The original analysis used four others:
boosted trees, decision trees, nearest neighbours, and a do-nothing baseline.
None has been ported or checked. Two of them wrap hyperparameter tuning
inside each split, adding randomness of their own, and the per-split results
from the original runs did not survive, so there is nothing local to compare
against. Each needs its own verification.

## Why we did not use verdict agreement as the standard

The obvious way to check a port is to ask whether both versions reach the
same significant-or-not verdict on each contrast. We tried that, and three
variations on it, before settling on the standard above:

| Candidate                                       | Result                         |
| ----------------------------------------------- | ------------------------------ |
| Verdicts agree, uncorrected                     | 3 of 4                         |
| Verdicts agree, Bonferroni correction           | 3 of 4, a different comparison |
| Verdicts agree, Benjamini-Hochberg correction   | 2 of 4                         |
| Each estimate falls inside the other's interval | 3 of 4                         |
| The two intervals overlap                       | 4 of 4                         |

Each fails for its own reason.

The R runs do not meet verdict agreement against each other. One February run
calls the 2019 Other-minus-Same contrast significant at 0.0196 while four
runs sharing its fold assignment call it null between 0.059 and 0.107. A
standard the reference cannot meet against itself cannot be required of a
port.

Which comparison disagrees also moves depending on how you correct for
testing several contrasts at once, and that correction is a judgement made
after seeing the data.

Estimate containment gets harder to satisfy as measurements improve, since
tighter intervals are easier to fall outside. A better port can fail where a
worse one passed, which is a strange property for a standard.

Interval overlap has the opposite defect. It is the most lenient of the
three, with poor power as an equivalence check (Schenker and Gentleman 2001),
and the most lenient candidate being the one that passes is not reassuring.

All of them also work on summaries: ten AUC numbers per cell, each condensing
about 1,800 predictions. Paired t-tests on overlapping cross-validation folds
understate variance (Dietterich 1998; Nadeau and Bengio 2003; Bengio and
Grandvalet 2004), so the p-values are optimistic on both sides.

The fix was to change the quantity instead of trying more tests:
compare the predicted probabilities directly, against a threshold fixed
before the comparison was computed. The t-tests remain the scientific output,
since they answer whether pooling helps, which is what SOAK exists to ask.

## What we still do not know

What changed the fold assignment between February and August. The draw
belongs to `mlr3resampling`, so a version change there is the obvious
candidate, but it is untested and February's assignment is not recorded
anywhere we can read.

Why the level shift exists. Penalty selection is the suspect, untested.

Whether any of this carries over to the other four models.

## What we got wrong along the way

An earlier draft led with a claim that a package update shifted the R results
by 0.0073 mean absolute AUC, larger than the effects the study reports. That
was the most striking thing we had and it was an artifact of comparing runs
that divided the data differently.

Three smaller corrections:

- The reference was switched from a February run to an August one. The change
  was right; the reason given was wrong. It fixed a fold mismatch, not a
  package mismatch.
- An interval half-width was computed as the upper bound instead of half the
  width, which overstated one ratio.
- Contrast tables in an earlier draft mixed a ridge run and a lasso run.
  Every figure in this version comes from the lasso run named above.

Printouts kept from earlier will not match this document.

## Reproducing

    NSCH_SOAK_REFERENCE=$REPRO/results/seed-variation/NSCH_seed1.csv \
      uv run python analyses/run_glmnet_replication.py --full-only --lasso \
      --l1-solver liblinear --intercept-scaling 100 \
      --out analyses/glmnet_replication_lasso_seed1_is100.csv \
      --save-predictions $REPRO/results/predictions/python_lasso_seed1_is100_predictions.csv

    uv run --with matplotlib python analyses/soak_ttests.py

    uv run python analyses/prediction_equivalence.py \
      --r-predictions      $REPRO/results/predictions/NSCH_seed1_predictions_repaired.csv \
      --python-predictions $REPRO/results/predictions/python_lasso_seed1_is100_predictions.csv \
      --python-auc         analyses/glmnet_replication_lasso_seed1_is100.csv

    uv run python analyses/cell_mean_distances.py --reproduce-dir $REPRO \
      --registry $PAPER/data_Classif_batchmark_registry.csv

    uv run python analyses/diagnose_fold_relabelling.py --reproduce-dir $REPRO

`$REPRO` is the reproduce-soak-nsch checkout and `$PAPER` the
cv-same-other-paper one. The R runs come from `NSCH_seed_variation.R`,
`NSCH_unseeded_check.R` and `NSCH_save_predictions.R`, about 25 minutes each
on eight cores.
