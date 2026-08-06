# Why the Python results differ from the R results

Chris Reger, 6 August 2026. Working note for the ASD3 methods paper.

## Short version

Everything upstream of model fitting matches exactly: rows, folds, train and
test membership on all 60 splits, downsample counts on all 40 downsampled
splits, all checked index for index against the archived R run.

The fitted models differ, and the size of that difference only means
something once you know how much the R analysis moves against itself. It
turns out to move quite a lot, and for a reason worth knowing: **the
`mlr3learners` version shifts the results by about 0.0073 mean absolute
AUC**, while the `cv.glmnet` inner cross-validation seed shifts them by
about 0.0005.

Against R runs on the same package build we now use, the Python port sits at
**0.0021**. That is four times the floor set by runs that differ only by
machine or seed, and about a third of the distance between two versions of R.

None of this is small next to the effects being measured. The SOAK contrasts
run around 0.0017, so build differences alone exceed the signal. Two of four
significance verdicts flip between R runs, before Python enters at all.

## What is identical

| | Checked against |
|---|---|
| 46,010 rows, 364 features | `data_Classif/NSCH_autism.csv` |
| Fold assignment | `nsch_autism_folds.csv`, used directly |
| Train and test rows, 60 full splits | `nsch_autism_iterations_long.csv`, exact |
| Downsample sizes and per-stratum counts, 40 splits | same, exact |

Row membership of downsampled train sets does not match and never could,
since R's `sample()` and NumPy's generator are different machines. The rule
matches; the draw does not. Only full splits feed the model comparison.

## How much does R disagree with itself?

Ten R result files now exist for the same analysis: five from February and
March under different backends and machines, one from batchtools, three run
on 6 August varying only the `cv.glmnet` seed, and one on 6 August with the
seed left unset. Clustering all 45 pairwise distances gives clear structure
rather than a spread.

| | Runs | Within-cluster mean |
|---|---|---|
| Cluster 1, Feb/March | local, local_desktop, local_laptop, mpi, proj | 0.000485 |
| Cluster 2, 6 August | seed1, seed2, seed3, unseeded | 0.000373 |
| Cluster 3 | batchtools | single member |

| Between | Mean absolute AUC difference |
|---|---|
| Cluster 1 to cluster 2 | 0.007299 |
| Cluster 1 to batchtools | 0.007908 |
| Cluster 2 to batchtools | 0.008437 |
| **Python to cluster 2** | **0.002095** |
| Python to cluster 1 | 0.007114 |

Three things fall out.

**The inner seed is worth about 0.0005.** Runs differing only in that seed
agree as closely as runs on different machines.

**Leaving the seed unset is identical to setting it to 1.** `NSCH_unseeded`
and `NSCH_seed1` agree on all 60 AUCs to the last bit. The learner defaults
to 1, which means the February runs were effectively seeded all along and
the seeding change explains nothing.

**The package version is worth about 0.0073.** With seeding ruled out, the
one remaining difference between February's runs and August's is
`mlr3learners`, updated to 0.14.0.9000 (the `tdhock/mlr3learners@cv_glmnet_seed`
fork) when we needed the seed parameter exposed. That is the whole
cluster separation, and it is larger than the effects the study reports.

Worth telling Toby about, independent of anything to do with our port.

## Where the port actually sits

Against R runs on the build we are now using, Python is 0.0021 away. That is
four times the within-cluster floor of 0.00047, so there is real
implementation distance and we are not at the limit of what is achievable.
But it is roughly a third of the 0.0073 that separates two versions of the
reference from each other.

Put plainly: our reimplementation agrees with the current R analysis more
closely than the current R analysis agrees with its own February self.

## What differs, and why

**Penalty selection, which is most of it.** `cv.glmnet` derives a lambda
path from the data, roughly 100 values, runs an internal 10-fold
cross-validation, scores by binomial deviance, and predicts at
`lambda.1se`. We run 5-fold cross-validation over a fixed 60-point grid of
`C`, score by log loss, and apply the same one-standard-error rule. Different
candidates, different inner folds, different scoring scale. These are two
models chosen by similar but distinct procedures, not one model computed
twice.

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

Paired t on 9 degrees of freedom against `NSCH_proj` (cluster 1):

| Test subset | Train source | Mean AUC difference | 95% interval |
|---|---|---|---|
| 2019 | Same | +0.00016 | (−0.00498, +0.00530) |
| 2019 | Other | −0.00081 | (−0.00563, +0.00402) |
| 2019 | All | −0.00056 | (−0.00563, +0.00451) |
| 2020 | Same | +0.00062 | (−0.00676, +0.00799) |
| 2020 | Other | +0.00157 | (−0.00658, +0.00972) |
| 2020 | All | +0.00069 | (−0.00646, +0.00784) |

No systematic lean, every interval inside a plus-or-minus 0.01 margin. These
would tighten against a cluster 2 reference; that comparison has not been
rerun.

There is also an outside check. Hocking et al. report mean AUC on the 2020
test subset of 0.9670 training on All and 0.9658 on Same. We get 0.9679 and
0.9664; `NSCH_proj` gives 0.9672 and 0.9658. The paper rounds to four
places, so agreement near a thousandth is the most this can show.

## The criterion problem

Toby's two-sided paired t on 9 degrees of freedom, per test subset, on both
implementations:

| Contrast | Subset | R estimate | R p | Python estimate | Python p |
|---|---|---|---|---|---|
| All − Same | 2019 | +0.00311 | 0.0003 | +0.00239 | 0.0326 |
| All − Same | 2020 | +0.00139 | 0.0068 | +0.00146 | 0.0037 |
| Other − Same | 2019 | +0.00176 | 0.0619 | +0.00079 | 0.5531 |
| Other − Same | 2020 | −0.00168 | 0.0365 | −0.00073 | 0.1654 |

Three of four verdicts match. Correcting for multiplicity, which we should,
since this is four tests and the full analysis will run a couple of hundred:

| Adjustment | Verdicts agreeing | Which one disagrees |
|---|---|---|
| None | 3 of 4 | Other − Same, 2020 |
| Bonferroni | 3 of 4 | **All − Same, 2019** |
| Benjamini-Hochberg | 2 of 4 | both of the above |

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

## What I would propose

Report the paired t-test p-values, since they answer the scientific question
and they are the method the SOAK paper prescribes. Judge whether the port
reproduces the analysis on whether contrast estimates agree within their
intervals.

On that criterion the port passes 4 of 4: in every comparison R's estimate
falls inside Python's interval and Python's falls inside R's. One is close.
On 2020 Other-minus-Same, R's −0.00168 clears the edge of Python's interval,
which runs to −0.00182, by fourteen hundred-thousandths.

**And pin the package version.** A build change moves these results further
than the effects being reported. Whatever tolerance we adopt is meaningless
without recording which `mlr3learners` produced the reference.

## What we do not know yet

Why batchtools sits apart from both clusters. It differs from cluster 1 by
0.0079 despite sharing February's build, so execution backend contributes
something on its own.

Whether the same structure holds for other learners. Everything here is
`cv_glmnet`. Boosted trees have their own randomization and may behave
differently.

Whether the per-split intervals tighten against a cluster 2 reference. They
were computed against `NSCH_proj` and have not been rerun.

## Reproducing

    uv run python analyses/run_glmnet_replication.py --full-only \
        --out analyses/glmnet_replication_grid60.csv
    uv run --with matplotlib python analyses/soak_ttests.py \
        --results analyses/glmnet_replication_grid60.csv --auc-col auc_1se
    uv run python analyses/soak_criteria.py \
        --results analyses/glmnet_replication_grid60.csv --auc-col auc_1se
    uv run python analyses/r_vs_r.py \
        --reproduce-dir /path/to/reproduce-soak-nsch

The R runs come from `NSCH_seed_variation.R` and `NSCH_unseeded_check.R` in
the reproduce-soak-nsch checkout, about 25 minutes each on 8 cores.

The 60-split Python run takes about eight minutes with ridge and L-BFGS. The
lasso equivalent took eleven and a half hours on the same machine, which is
worth recording: glmnet's coordinate descent handles this problem in minutes
where scikit-learn's saga solver needs most of a day.
