# Why the Python results differ from the R results

Chris Reger, 6 August 2026. Working note for the ASD3 methods paper.

## Short version

Everything upstream of model fitting matches exactly: rows, folds, train and
test membership on all 60 splits, downsample counts on all 40 downsampled
splits, all checked index for index against the archived R run.

The fitted models differ. How much they differ depends almost entirely on
which R run you compare against, and that turned out to be the most
important thing we learned. **The `mlr3learners` build shifts the R results
by about 0.0073 mean absolute AUC**, while the `cv.glmnet` inner seed shifts
them by about 0.0005. For a while we were measuring our port against a
reference from an older build and attributing the difference to ourselves.

Against R runs on the build we actually use, the port sits at **0.0021**,
with a typical interval half-width of 0.0018 against contrasts of about
0.0016. Roughly one to one. Against the older build the same comparison gave
0.0071 and a ratio near four to one.

That is close, but not close enough to disappear. One contrast estimate now
falls outside its counterpart's interval, and one cell shows a small
difference at a raw p of 0.012, which does not survive correction across the
six cells tested. Both only became visible once the reference stopped
contributing 0.007 of its own noise.

## What is identical

|                                                    | Checked against                          |
| -------------------------------------------------- | ---------------------------------------- |
| 46,010 rows, 364 features                          | `data_Classif/NSCH_autism.csv`           |
| Fold assignment                                    | `nsch_autism_folds.csv`, used directly   |
| Train and test rows, 60 full splits                | `nsch_autism_iterations_long.csv`, exact |
| Downsample sizes and per-stratum counts, 40 splits | same, exact                              |

Row membership of downsampled train sets does not match and never could,
since R's `sample()` and NumPy's generator are different machines. The rule
matches; the draw does not. Only full splits feed the model comparison.

## How much does R disagree with itself?

Ten R result files now exist for the same analysis: five from February and
March under different backends and machines, one from batchtools, three run
on 6 August varying only the `cv.glmnet` seed, and one on 6 August with the
seed left unset. Clustering all 45 pairwise distances gives clear structure
rather than a spread.

|                            | Runs                                          | Within-cluster mean |
| -------------------------- | --------------------------------------------- | ------------------- |
| Cluster 1, Feb/March build | local, local_desktop, local_laptop, mpi, proj | 0.000485            |
| Cluster 2, 6 August build  | seed1, seed2, seed3, unseeded                 | 0.000373            |
| Cluster 3                  | batchtools                                    | single member       |

| Between                 | Mean absolute AUC difference |
| ----------------------- | ---------------------------- |
| Cluster 1 to cluster 2  | 0.007299                     |
| Cluster 1 to batchtools | 0.007908                     |
| Cluster 2 to batchtools | 0.008437                     |
| **Python to cluster 2** | **0.002095**                 |
| Python to cluster 1     | 0.007114                     |

Three things fall out.

**The inner seed is worth about 0.0005.** Runs differing only in that seed
agree as closely as runs on different machines.

**Leaving the seed unset is identical to setting it to 1.** `NSCH_unseeded`
and `NSCH_seed1` agree on all 60 AUCs to the last bit. The learner defaults
to 1, which means February's runs were effectively seeded all along and the
seeding change explains nothing.

**Something between February and August is worth about 0.0073, and it is
not the seed.** The known change is `mlr3learners`, updated to 0.14.0.9000
(the `tdhock/mlr3learners@cv_glmnet_seed` fork) when we needed the seed
parameter exposed, and that is the likely cause. But the February runs also
differ in machine, and possibly in R version and linear-algebra library, so
the attribution rests on one confounded comparison where the seed rested on
a clean one. The test that settles it is installing the February
`mlr3learners` and rerunning once; that run is planned and cheap.

Whichever component it is, the size alone is worth telling Toby about,
independent of anything to do with our port.

## Which reference to use

Everything below compares against `NSCH_seed1`, a cluster 2 run on the build
we have installed. Earlier drafts of this note used `NSCH_proj`, from
February. Switching cut the measured implementation gap by a factor of
three and a half, which is the clearest possible illustration of why the
build has to be pinned before any tolerance is stated.

The archival figures against `NSCH_proj` are kept in
`analyses/glmnet_replication_grid60.csv` and noted below where they differ
materially. The Python fits are identical in both files; only the reference
column changes.

## What differs, and why

**Penalty selection, which is most of it.** `cv.glmnet` derives a lambda
path from the data, roughly 100 values, runs an internal 10-fold
cross-validation, scores by binomial deviance, and predicts at
`lambda.1se`. We run 5-fold cross-validation over a fixed 60-point grid of
`C`, score by log loss, and apply the same one-standard-error rule.
Different candidates, different inner folds, different scoring scale. These
are two models chosen by similar but distinct procedures, not one model
computed twice.

**Optimizer.** glmnet uses cyclical coordinate descent. We use L-BFGS. At an
identical penalty the fitted coefficients still differ slightly.

**Parameterization.** glmnet minimizes `(1/n) * deviance + lambda *
penalty`; scikit-learn minimizes `C * loss + penalty`. So lambda is roughly
`1/(nC)`, and the grids align only approximately.

Two candidate explanations were tested and ruled out. Penalty family is not
the cause: ridge, lasso, and a 60-point ridge grid put the 2019
All-minus-Same estimate within 0.0006 of each other. Grid coarseness is not
the cause either: going from 12 penalty values to 60 changed the selected
penalty by about 1.35x and moved nothing of substance.

## Size of the difference, split by split

Paired t on 9 degrees of freedom against `NSCH_seed1`:

| Test subset | Train source | Mean AUC difference | 95% interval             | p         |
| ----------- | ------------ | ------------------- | ------------------------ | --------- |
| 2019        | Same         | −0.00024            | (−0.00305, +0.00258)     | 0.854     |
| 2019        | Other        | −0.00108            | (−0.00374, +0.00158)     | 0.383     |
| 2019        | All          | −0.00085            | (−0.00282, +0.00111)     | 0.352     |
| 2020        | Same         | +0.00071            | (−0.00089, +0.00232)     | 0.340     |
| 2020        | Other        | **+0.00205**        | **(+0.00056, +0.00353)** | **0.012** |
| 2020        | All          | +0.00055            | (−0.00057, +0.00167)     | 0.297     |

Five of six show no detectable difference. The sixth, 2020 Other, has an
interval excluding zero at a raw p of 0.012. Six cells were tested, so under
either Bonferroni or Benjamini-Hochberg that becomes 0.074: suggestive, not
established. A document that argues for multiplicity correction does not get
to skip it for its own findings.

Worth watching rather than concluding, then. If it is real it is about two
thousandths of AUC, which changes no one's conclusions, but it would be the
first sign of a systematic implementation difference rather than scatter,
and it only became visible once the reference tightened. The prediction-level
comparison planned below will settle it with far more data than ten folds.

There is also an outside check. Hocking et al. report mean AUC on the 2020
test subset of 0.9670 training on All and 0.9658 on Same. We get 0.9679 and
0.9664; `NSCH_seed1` gives 0.9674 and 0.9657. The paper rounds to four
places, so agreement near a thousandth is the most this can show.

## The criterion problem

Toby's two-sided paired t on 9 degrees of freedom, per test subset, on both
implementations:

| Contrast     | Subset | R estimate | R p    | Python estimate | Python p |
| ------------ | ------ | ---------- | ------ | --------------- | -------- |
| All − Same   | 2019   | +0.00301   | 0.0015 | +0.00239        | 0.0326   |
| All − Same   | 2020   | +0.00163   | 0.0006 | +0.00146        | 0.0037   |
| Other − Same | 2019   | +0.00164   | 0.0903 | +0.00079        | 0.5531   |
| Other − Same | 2020   | −0.00206   | 0.0139 | −0.00073        | 0.1654   |

Three of four verdicts match. Correcting for multiplicity, which we should,
since this is four tests and the full analysis will run a couple of hundred:

| Adjustment         | Verdicts agreeing | Which one disagrees  |
| ------------------ | ----------------- | -------------------- |
| None               | 3 of 4            | Other − Same, 2020   |
| Bonferroni         | 3 of 4            | **All − Same, 2019** |
| Benjamini-Hochberg | 2 of 4            | both of the above    |

The count barely moves. The comparison does. Which result disagrees depends
on a correction chosen after seeing the data.

**And R fails this criterion against itself.** Across the ten R runs, two of
four contrasts get different verdicts depending on which run you use. The
February `mpi` run calls 2019 Other-minus-Same significant at 0.0196 while
four sibling runs that agree with it to 0.0005 on AUC call it null between
0.059 and 0.107. One August run landed at 0.0502, two ten-thousandths from
flipping.

Verdict agreement is not a standard the reference analysis meets against
itself. It cannot reasonably be required of a port.

### Why it fails: the two scales

|                                             | Against cluster 2 | Against cluster 1 |
| ------------------------------------------- | ----------------- | ----------------- |
| Typical SOAK contrast under test            | 0.00163           | 0.00157           |
| Typical R-versus-Python interval half-width | 0.00179           | 0.00614           |
| Ratio                                       | **1.1x**          | 3.9x              |

Both figures are medians over both implementations and all six
subset-by-source cells. Against the current build the implementation gap is
about the same size as the effect being tested. Against the February build
it was four times larger, which is what made a p-value near 0.05 a coin
toss: 2019 All-minus-Same came out at 0.0544 under one configuration and
0.0326 under another, from estimates one hundred-thousandth apart.

One to one is much better than four to one. It is still not a comfortable
place to be running significance tests from.

## Where this leaves the criterion question

Report the paired t-test p-values, since they answer the scientific question
and they are the method the SOAK paper prescribes. The open question is what
standard the port should be held to, and this work has now tried three
candidates on fold-level AUC:

| Candidate                                       | Result against `NSCH_seed1`    |
| ----------------------------------------------- | ------------------------------ |
| Verdicts agree, uncorrected                     | 3 of 4                         |
| Verdicts agree, Bonferroni                      | 3 of 4, a different comparison |
| Verdicts agree, Benjamini-Hochberg              | 2 of 4                         |
| Each estimate falls inside the other's interval | 3 of 4                         |
| The two intervals overlap                       | 4 of 4                         |

None of these is fit to be the standard, and each fails differently.
Verdict agreement is not met by the R runs against each other, and which
comparison fails moves with the multiplicity adjustment. Estimate
containment gets harder to satisfy as intervals tighten, so a
better-measured comparison can fail where a noisier one passed; it did
exactly that when the reference switched, on 2020 Other-minus-Same, where
R's −0.00206 now sits just outside Python's interval after sitting just
inside the February one by fourteen hundred-thousandths. Interval overlap
has the opposite defect: it is the weakest of the three, known to have poor
power as an equivalence check (Schenker and Gentleman 2001), and it is not
reassuring that the most lenient candidate is also the one that passes.

There is a structural problem underneath all three: every candidate was
evaluated after its results were visible, in a document that argues
conclusions should not depend on choices made after seeing the data. And
all three operate on fold-level AUC, ten numbers per cell summarising
1,819 predictions each, with the added caveat that paired t-tests on
overlapping cross-validation folds understate variance (Dietterich 1998;
Nadeau and Bengio 2003; Bengio and Grandvalet 2004), so the p-values above
are optimistic on both sides.

The resolution is to change the quantity, not to keep auditioning tests.
The plan of record: compare the predicted probabilities themselves on the
60 full splits, same test rows both sides, against a margin fixed in
writing before that comparison is computed. The fold-level results in this
document then stand as what they are, a characterisation of how the
summaries behave, rather than as the replication standard.

**And pin the build regardless.** A package change moves these results four
times further than our own implementation does. Whatever margin is adopted
is meaningless without recording which `mlr3learners` produced the
reference.

## The comparison against the pre-registered margins

Run on 7 August against `NSCH_seed1`, using our lasso fits with liblinear at
`intercept_scaling=100`. The margins are the ones in
`docs/equivalence-margin.md`, committed the day before the comparison script
was written.

R's `cv_glmnet` predicts at `lambda.1se`, so that is the gated column.
`lambda.min` is reported beside it.

| Quantity                                 | Observed | Margin        |          |
| ---------------------------------------- | -------- | ------------- | -------- |
| Spearman of held-out scores, worst split | 0.94057  | at least 0.95 | **fail** |
| Probability MAD, worst split             | 0.006554 | at most 0.01  | pass     |
| Fold-level AUC, mean absolute            | 0.001271 | at most 0.002 | pass     |

Fifty-eight of sixty splits clear every gated check. Two do not.

### The two failures are one fitted model

They are 2019 "same" fold 7 at 0.94057 and 2020 "other" fold 7 at 0.94106.
Those two cells train on identical rows, and their R coefficients here differ
by exactly zero, so this is one fit scored on two test subsets rather than
two independent failures.

Nothing about the fit stands out. It holds out 1,821 children, 55 of whom
have the outcome, a rate of 0.0302 that matches folds 2, 6, 8 and 9 exactly.
It keeps 19 features, as do folds 3, 5, 6 and 10, three of which pass
comfortably.

### What Spearman is measuring here

The children the two implementations rank most differently are all true
negatives at probabilities below 0.007:

| row_id | outcome | R       | Python  | rank displacement |
| ------ | ------- | ------- | ------- | ----------------- |
| 1587   | 0       | 0.00345 | 0.00087 | 972               |
| 1252   | 0       | 0.00216 | 0.00236 | 732               |
| 13409  | 0       | 0.00216 | 0.00235 | 728               |
| 1439   | 0       | 0.00216 | 0.00234 | 726               |

The three consecutive rows are the giveaway. R returns exactly 0.00216 for
all of them and Python returns three slightly different values, so R has
produced a tie and Python has broken it. Spearman gives tied observations
their average rank, which means any ordering imposed on them counts as a
large displacement. On this split 828 of 1,821 children move more than 100
ranks.

At a 3% base rate, 1,766 of those children are ones the model is confident
about. Ranking them against each other says nothing about whether the two
implementations agree on who is at risk, and that is where nearly all of
Spearman's power goes. The probability MAD for the same split is 0.0058,
well inside its margin.

### What this means

The margin stands. It was fixed before the comparison existed, and moving it
now to accommodate a result we have already seen would forfeit the whole
point of fixing it.

The reading we can defend is that the port reproduces the R analysis on 58 of
60 splits, and that the criterion which failed was a poor choice for a rare
outcome. That is a fault in how we designed the check rather than evidence
about the port, and the other two gated quantities pass on all 60 splits
including this one.

Three changes for the next pre-registration, to be fixed in advance again
rather than applied backwards to this one:

- compute rank agreement over the children the model ranks highest, not over
  the full held-out set, because the ordering of confident negatives is
  uninformative and swamps the statistic;
- make probability MAD primary rather than secondary, since it is the
  quantity that answers whether two implementations produce the same
  predictions;
- decide how to handle ties in the reference, because a coarser probability
  resolution on one side penalises the other for having a finer one.

`lambda.min` points the same way from a different angle. Its rank agreement
is much worse, averaging about 0.93 with a worst split of 0.866, while its
probability MAD stays near 0.008. Gating on `lambda.1se`, which is what
`cv_glmnet` predicts at, was the right call and now has evidence behind it.

### One offset we cannot yet explain

Python's fold-level AUC runs about 0.0012 above R's at `lambda.1se`,
consistently, with a paired t of +5.56 across the 20 fold clusters. That is
half the margin, so it passes, but it is systematic rather than noise.

It is not a bias in the probabilities themselves. The mean signed difference
is −0.0001, near enough to zero. Neither implementation is producing
uniformly higher predictions; they are ordering children slightly
differently, and the difference happens to favour ours.

Penalty selection is the obvious suspect. `cv.glmnet` searches a lambda path
derived from the data, roughly 100 values, while we search a fixed 60-point
grid of `C` with different inner folds and a different scoring loss. A grid
that tends to land on a marginally better-performing penalty would produce
exactly this pattern. Untested so far.

## What we do not know yet

Why 2020 Other specifically. It is the one cell whose interval excludes
zero, though not after multiplicity correction, and nothing about the
penalty-selection story predicts a single cell rather than a general shift.
The planned coefficient and prediction comparison on those ten splits will
say whether it is real.

Why batchtools sits apart from both clusters. It differs from cluster 1 by
0.0079 despite sharing February's build, so execution backend contributes
something on its own.

Whether the same structure holds for other learners. Everything here is
`cv_glmnet`. Boosted trees have their own randomization and may behave
differently.

## Provenance of these numbers

Every figure above comes from the scripts in `analyses/`, run against the
files listed below. Three corrections were made during the work, so
printouts kept from earlier will not match:

- The reference was switched from `NSCH_proj` (February build) to
  `NSCH_seed1` (current build). This is the change that matters; it cut the
  measured implementation gap by a factor of three and a half.
- The implementation half-width was originally taken as the upper interval
  bound rather than `(hi - lo) / 2`. Those coincide only when the mean sits
  at zero, and the error overstated the ratio.
- The two scripts computed "typical contrast" over different sets, one using
  both implementations and one using R alone, and printed different ratios
  under the same label. Both now median over both.

The scripts are the source; this document records what they produced.

## Reproducing

    NSCH_SOAK_REFERENCE=/path/to/reproduce-soak-nsch/results/seed-variation/NSCH_seed1.csv \
      uv run python analyses/run_glmnet_replication.py --full-only \
      --out analyses/glmnet_replication_seed1.csv
    uv run --with matplotlib python analyses/soak_ttests.py \
        --results analyses/glmnet_replication_seed1.csv
    uv run python analyses/soak_criteria.py \
        --results analyses/glmnet_replication_seed1.csv
    uv run python analyses/r_vs_r.py \
        --reproduce-dir /path/to/reproduce-soak-nsch

The R runs come from `NSCH_seed_variation.R` and `NSCH_unseeded_check.R` in
the reproduce-soak-nsch checkout, about 25 minutes each on 8 cores.

The 60-split Python run takes about eight minutes with ridge and L-BFGS. The
lasso equivalent took eleven and a half hours on the same machine, which is
worth recording: glmnet's coordinate descent handles this problem in minutes
where scikit-learn's saga solver needs most of a day.
