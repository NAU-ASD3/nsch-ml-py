# Why the Python results differ from the R results, and what to do about it

Chris Reger, 6 August 2026. Working note for the ASD3 methods paper.

## The short answer

Everything upstream of model fitting is exact. The rows, the folds, the
train and test membership on all 60 splits, the downsample counts on all 40
downsampled splits: all verified index for index against the archived R
instance. Nothing there is approximate.

The differences come entirely from fitting the penalized model, and they are
small. Per test fold, the Python and R AUCs sit within about seven
thousandths of each other, with a mean difference near zero in all six
subset-by-source cells.

That sounds like a clean result, and in one sense it is. The problem is that
the effects SOAK exists to measure are also small. The All-minus-Same and
Other-minus-Same contrasts on this dataset run around 0.0016. So the gap
between two reasonable implementations is roughly four times the size of the
signal we are testing for. Two of our four significance verdicts flip as a
result, even though every estimate agrees.

## What is identical

| | Verified against |
|---|---|
| 46,010 rows, 364 features | `data_Classif/NSCH_autism.csv` |
| Fold assignment | `nsch_autism_folds.csv`, used directly |
| Train and test rows, 60 full splits | `nsch_autism_iterations_long.csv`, exact |
| Downsample sizes and per-stratum counts, 40 splits | same, exact |

Row membership of the downsampled train sets does not match and cannot,
since R's `sample()` and NumPy's generator are different. The rule matches;
the draw does not. Only the full splits are used for the model comparison,
so this does not enter.

## What differs, in rough order of how much it matters

**Penalty selection.** `cv.glmnet` builds its own lambda path from the data,
about 100 values, runs an internal 10-fold cross-validation, scores by
binomial deviance, and predicts at `lambda.1se`. We run 5-fold
cross-validation over a fixed 60-point grid of `C`, score by log loss, and
apply the same one-standard-error rule to that grid. Different candidate
penalties, different inner folds, different scoring scale. The two sides are
not the same model computed twice; they are two models selected by similar
but distinct procedures.

**Optimizer.** glmnet uses cyclical coordinate descent with its own
convergence criteria. We use L-BFGS. Even at an identical penalty the fitted
coefficients differ slightly.

**Parameterization.** glmnet minimizes `(1/n) * deviance + lambda *
penalty`; scikit-learn minimizes `C * loss + penalty`. So `lambda` is
roughly `1/(nC)` and our grid aligns only approximately with theirs.

**Inner cross-validation randomization.** Neither side's inner folds are
matched to the other's. They cannot be, for the same reason the downsample
draws cannot.

Two things worth ruling out, because both were checked. Alpha is not the
cause: ridge, lasso, and a much finer ridge grid all put the Python
All-minus-Same estimate for 2019 within 0.0006 of each other. Grid
coarseness is not the cause either: going from 12 penalty values to 60
changed the selected penalty by about 1.35x across splits and moved nothing
of substance.

## What the differences look like

Against the archived R run, per subset and train source, paired t on 9
degrees of freedom:

| Test subset | Train source | Mean AUC difference | 95% interval |
|---|---|---|---|
| 2019 | Same | +0.00016 | (−0.00498, +0.00530) |
| 2019 | Other | −0.00081 | (−0.00563, +0.00402) |
| 2019 | All | −0.00056 | (−0.00563, +0.00451) |
| 2020 | Same | +0.00062 | (−0.00676, +0.00799) |
| 2020 | Other | +0.00157 | (−0.00658, +0.00972) |
| 2020 | All | +0.00069 | (−0.00646, +0.00784) |

No systematic bias in either direction. Every interval sits inside a
plus-or-minus 0.01 margin.

There is also an external check. Hocking et al. report mean AUC on the 2020
test subset of 0.9670 training on All and 0.9658 training on Same. We get
0.9679 and 0.9664. The paper rounds to four places, so agreement near a
thousandth is as much as this comparison can demonstrate, and we have it.

## Where the trouble is

Running Toby's two-sided paired t on 9 degrees of freedom, per test subset,
on both implementations:

| Contrast | Subset | R estimate | R p | Python estimate | Python p |
|---|---|---|---|---|---|
| All − Same | 2019 | +0.00311 | 0.0003 | +0.00239 | 0.0326 |
| All − Same | 2020 | +0.00139 | 0.0068 | +0.00146 | 0.0037 |
| Other − Same | 2019 | +0.00176 | 0.0619 | +0.00079 | 0.5531 |
| Other − Same | 2020 | −0.00168 | 0.0365 | −0.00073 | 0.1654 |

Signs agree everywhere. Estimates are close. Three of the four significance
verdicts match; the fourth, 2020 Other-minus-Same, has R rejecting at 0.0365
and Python not rejecting at 0.1654.

Now correct for multiplicity, which we should, since these are four tests
and the full analysis will run a couple of hundred. Under Bonferroni at four
tests, R's 0.0365 no longer clears the bar either, so that disagreement
disappears and 2019 All-minus-Same becomes the one that differs. Under
Benjamini-Hochberg the count drops further.

The disagreement does not go away under correction. It moves. Which
comparison disagrees depends on which correction we choose, and choosing
one is a judgement call we make after seeing the data.

## The criterion I would propose

Verdict agreement fails as a replication criterion for a reason that has
nothing to do with the quality of the port. When the effect is 0.0016, the
implementation gap is 0.0067, and there are ten folds, a p-value near 0.05
is a coin toss. We saw exactly that: 2019 All-minus-Same came out at 0.0544
under one configuration and 0.0326 under another, from estimates that
differed by one hundred-thousandth.

Estimate agreement holds through all of it. In every configuration we ran,
R's contrast estimate sits inside Python's confidence interval and Python's
sits inside R's. That is stable, it does not depend on a multiplicity
choice, and it is the claim the data actually support.

So: report the t-test p-values as Toby's method prescribes, since they are
the right tool for the scientific question. Judge the replication on
whether the contrast estimates agree within their intervals. Those are
different questions and they deserve different instruments.

## The calibration experiment we are missing

The honest benchmark is not "how close is Python to R." It is "how close is
R to itself." `cv.glmnet`'s inner cross-validation is randomized, so
re-running the R analysis under a different seed produces a different set of
selected penalties and a different set of AUCs. If R-versus-R variation
across seeds is comparable to the R-versus-Python variation we measured,
then the port is at the floor and no amount of further work will close the
gap.

That run is cheap on the R side and would settle the question. I would like
to do it before the methods paper states a tolerance.

## One thing to flag about the manuscript

Vince's SOAK section does not use a significance test. It reports that
standard deviation bands overlap and concludes from that overlap that a gap
is not statistically significant. Applying the paired t-test from the SOAK
paper is therefore an addition to the published analysis rather than a
reproduction of it, and it may reach different conclusions on temporal
drift and on the AI/AN fairness comparison. Those differences would be
findings, not replication failures, and we should say so plainly when they
come up.

## Reproducing

    uv run python analyses/run_glmnet_replication.py --full-only \
        --out analyses/glmnet_replication_grid60.csv
    uv run --with matplotlib python analyses/soak_ttests.py \
        --results analyses/glmnet_replication_grid60.csv --auc-col auc_1se
    uv run python analyses/soak_criteria.py \
        --results analyses/glmnet_replication_grid60.csv --auc-col auc_1se

The fixture and the R reference live outside the repository; both paths come
from environment variables. The 60-split run takes about eight minutes with
ridge and L-BFGS. The lasso equivalent took eleven and a half hours on the
same machine, which is worth recording on its own: glmnet's coordinate
descent handles this problem in minutes where scikit-learn's saga solver
needs most of a day.
