# The extension analysis plan

Chris Reger, 12 August 2026.

This fixes the analysis choices for the three service outcomes before any
model has been fitted against them. The git timestamp is the point. Every
number below describes the data, the fold assignments, or a decision; none of
it is a result, because no result exists yet.

Like `equivalence-margin.md`, this document is append-only once committed.
Choices are amended by dated addenda at the bottom, never by editing the text
above. A plan that can be quietly rewritten after the results arrive is not a
plan.

The replication that licenses all of this is written up in
`replication-equivalence.md`. The short version: the Python port reproduces
the published R analysis, and the machinery below is the same machinery,
pointed at new outcomes.

## Which paper this serves

This matters enough to state before anything else, because the analysis below
is easy to mistake for something it is not.

**This plan serves the methods paper.** Its output is a set of comparisons
between training strategies, which answer the question SOAK exists to answer:
can survey periods be pooled when fitting a model, or does each need its own?
That is a question about cross-validation design.

**It does not, on its own, serve the outcomes manuscript.** A result of the
form "training on all periods beats training on one by 0.004 AUC" does not
become a sentence about children's access to care. The outcomes manuscript
needs survey-weighted prevalence estimates with confidence intervals,
adjusted odds ratios from a design-based model, and defensible statements
about which characteristics are associated with unmet need. None of those are
produced here.

What this pass hands to the outcomes manuscript is narrower and worth naming
precisely:

1. Whether periods can be pooled, which determines whether that manuscript
   analyses 2016 to 2023 as one population or period by period.
2. Validated, documented, checksummed data and fold assignments for the
   outcomes it will use.
3. Unweighted descriptive prevalence by period, as context rather than as a
   finding.

The outcomes manuscript's own analysis needs its own pre-registration, and
this document is not it. Writing that is a separate task and it is on the
critical path in a way the runs below are not.

## What this pass produces

Stated so that scope creep is visible if it happens.

- Per-split results for each task: AUC, accuracy, percent error, Brier score,
  calibration slope, with the fitted penalty and the training and test sizes.
- Per-child held-out predictions for every split, which is what makes any
  later metric computable without refitting.
- One confirmatory contrast per outcome, and a set of exploratory ones.
- Prevalence by period and by population, unweighted.
- A figure per outcome in the idiom of the replication figures: one point per
  train/test split, mean and standard deviation, no bars.

## What is being predicted

Three outcomes, all from the National Survey of Children's Health.

**Foregone care.** A child is a positive case when there was a time in the
past twelve months they needed health care and did not receive it. This is
`k4q27` in survey coding, and `Needed_Health_Care_Not_Received_Yes` on the
full-population matrix.

**Emergency department use.** Two definitions, both analysed. The survey
records emergency room visits in bands rather than as a count, and the
matrices carry indicator columns for the lowest bands, so the higher band is
the rows where every indicator is zero.

- **Any use**: one or more visits in the past twelve months.
- **Repeat use**: two or more visits.

Running both is deliberate. Any use captures contact with emergency care at
all. Repeat use is a narrower construct, and my understanding from a brief
reading is that repeat visits are sometimes treated as an indicator that
routine care is not meeting a family's needs, on the reasoning that a single
visit is often unavoidable. I have not read enough to know how well
established that is or whether it is the convention in the autism services
literature, and the question is with Olivia. Until it is settled, both are
reported and neither is privileged.

**Behaviour therapy.** A child is a positive case when they received
behavioural treatment for autism. This question is asked only of children who
already have a diagnosis, so it exists for one population only.

### A coding change that both ED definitions survive

`hospitaler` carried three response levels through 2021 and four from 2022,
splitting the top band. The autism-subset matrix spans 2016 to 2023 and so
crosses that change.

Both definitions above are built from the lowest two levels, which are
identical in both codings: "none" and "1 time" mean the same thing before and
after 2022. Any definition reaching into the top band, such as four or more
visits, would be confounded with survey year and is excluded for that reason.

The three-level structure on the autism-subset matrix was checked directly.
The residual share by period runs 7.1%, 6.9%, 5.3%, 6.8%, with no
discontinuity at the 2022-23 period, and the three states partition every
period exactly with no row setting two indicators at once. The harmonisation
folded the new top levels together cleanly.

## Which populations

| Outcome | Full child population | Children with autism |
| --- | :---: | :---: |
| Foregone care | yes | yes |
| ED use, any | yes | yes |
| ED use, repeat | yes | yes |
| Behaviour therapy | no, the question is not asked | yes |

Seven analysis tasks: three on the full-population matrix and four on the
autism-subset matrix. Every table and figure produced from these must state
which population and which years it describes.

| Matrix | md5 | Span | Subsets | Tasks |
| --- | --- | --- | ---: | ---: |
| `NSCH_autism.csv` | `8c1e5a82e75d9e74eaa94a2dbe6c8a36` | 2019 to 2020 | 2 | 3 |
| `2016_2023_ServiceUse.csv` | `f2a64b712a7d56f3898e94c48fbf0c65` | 2016 to 2023 | 4 | 4 |

An eighth task, the full population across all survey years, requires a matrix
that does not exist and is out of scope here.

### Counts these will run on

Full population, 46,010 children, 364 features, 18,202 in 2019 and 27,808 in
2020.

| Outcome | Positives, 2019 | Positives, 2020 | Total |
| --- | ---: | ---: | ---: |
| Foregone care | 410 | 908 | 1,318 |
| ED use, any | 3,043 | 3,874 | 6,917 |
| ED use, repeat | 580 | 715 | 1,295 |

Autism subset, 6,088 children across four two-year periods, 300 columns.

| Period | Children | Foregone care | ED, any | ED, repeat | Behaviour therapy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2016-17 | 1,038 | 9.1% | 23.7% | 7.1% | 64.5% |
| 2018-19 | 1,021 | 10.3% | 21.2% | 6.9% | 61.9% |
| 2020-21 | 1,672 | 14.1% | 18.5% | 5.3% | 60.6% |
| 2022-23 | 2,357 | 13.1% | 22.1% | 6.8% | 63.1% |

## Features, and what is removed

For each task the feature set is the matrix's full set, minus identifier and
design columns, minus the outcome's own columns.

On the full-population matrix that means all 364 features except those listed
below. `survey_year` is the subset label and `y`, the autism diagnosis, is the
replication's outcome; both stay out, keeping the feature set identical to the
validated one. Adding autism diagnosis as a predictor is a sensible variant
for the manuscript and is recorded here as a variant, not folded in silently.

On the autism-subset matrix the excluded columns are `hhid`, `stratum`, `fwc`,
`fipsst`, `state` and `period`.

### Leak audit

Only an outcome's own columns are removed.

| Outcome | Removed |
| --- | --- |
| Foregone care | the `k4q27` column |
| ED use, either definition | both `hospitaler` indicator columns |
| Behaviour therapy | the `autismtreat` column |

Every header was read directly rather than reasoned about. Four neighbouring
variables were examined and all four stay in.

`k4q20r` was initially removed, on my belief that it counted all health care
visits, which would mean zero visits logically implies no emergency visits.
That was wrong. The 2024 `.do` file labels it "Preventive Visit, How Many
Times", and its distribution matches: about half of children have exactly one,
which is the annual well-child pattern rather than a total-utilisation one. A
child can have no preventive visits and several emergency visits, so there is
no implication in either direction. It stays, and it is not a marginal keep:
preventive care predicting emergency use is close to the substantive question
these outcomes exist to ask.

`k4q02_r=Hospital Emergency Room` records where a child usually goes when
sick. That is a distinct construct from counting visits actually made, and it
stays.

`k4q22_r` and `k4q24_r` carry "No, but this child needed to see..." levels for
mental health professionals and specialists. These are specific instances of
needed care not received, and `k4q27` is the global version of the same
construct, so the overlap is closer to implication than to correlation. They
stay in the primary analysis, because removing every correlate of an outcome
leaves nothing to predict from. **Pre-registered sensitivity variant:** the
foregone care tasks are also run with both columns removed, and both results
are reported. If they disagree materially, the variant is the one to believe.

`autismmed` sits in the same question battery as `autismtreat`. Medication and
behavioural therapy are separate treatments and a child may receive either
alone, so it stays.

### Missing data, which is not yet settled

The encoding of item nonresponse is inherited from the pipelines that built
these matrices and has not been verified.

What has been checked: all four outcome columns on the autism-subset matrix
contain no missing values, and the full-population outcome has none either.
So no case is dropped or imputed for lack of an outcome.

What has not been checked: how nonresponse in the features was encoded, and
whether it is distinguishable from a substantive "no". This matters more here
than it did for the replication. A family that does not answer questions about
unmet need may plausibly differ from one that answers "no", so nonresponse
could be related to the outcome rather than incidental to it.

**Required before any result from these runs is interpreted:** establish how
missingness is encoded in each matrix and record it as an addendum here. If
the encoding turns out to be lossy, that is a finding about the matrices and
it changes what these runs can support.

## Fold assignment

Each outcome gets its own assignment, drawn in Python by
`nsch_ml.soak.assign_folds`, stratified on (subset, outcome), ten folds, seed
1.

The assignment itself is not tracked in git. It is fully determined by the
matrix checksum, the outcome key, the seed and the fold count, and redrawing
it takes seconds, so what is tracked is the provenance record naming all four
plus the package versions in use. That is the reverse of the treatment given
to model results, which are tracked because reproducing them costs over an
hour apiece. Every results file names the provenance record its folds came
from, and a result whose folds cannot be redrawn from a tracked provenance
record is not reportable.

The splitter deals each (subset, outcome) cell across the folds round robin,
so per-cell positive counts differ by at most one. The draws for the three
full-population outcomes were checked and hold to that.

### A test that was written, fired, and retired

The original plan was to reuse the validated R-drawn fold assignment if the
new outcomes happened to be balanced across it, against a rule fixed in
advance: no (year, fold) cell below five positives, and no cell more than 20%
from its year's per-fold mean.

The rule fired. Foregone care in 2019 deviated by 26.8% and repeat ED use in
2019 by 25.9%.

It fired because it was mis-specified, not because the folds were unusual. The
deviation of a count across folds scales as one over the square root of the
number of positives, so a fixed percentage threshold flags rare outcomes and
passes common ones regardless of whether the assignment carries any outcome
structure. Comparing each observed deviation against what random allocation of
the same number of positives produces:

| Outcome and year | Positives | Observed | Chance median | Percentile |
| --- | ---: | ---: | ---: | ---: |
| Foregone care 2019 | 410 | 26.8% | 26.8% | 43rd |
| Foregone care 2020 | 908 | 18.5% | 17.8% | 51st |
| ED any 2019 | 3,043 | 9.1% | 10.0% | 38th |
| ED any 2020 | 3,874 | 10.0% | 8.7% | 69th |
| ED repeat 2019 | 580 | 25.9% | 22.4% | 73rd |
| ED repeat 2020 | 715 | 17.5% | 20.3% | 35th |

Every value sits in the middle of the chance distribution. The assignment
behaves exactly as an unstratified split does with respect to these outcomes,
which is what it is: it was stratified on the autism diagnosis, and on that it
deviates by 1.1% and 0.8%.

The threshold was **not** relaxed. The test was retired, because it measured
prevalence rather than balance, and the decision was moved to a basis that
does not need it: draw fresh folds stratified on each outcome. That is
strictly better than reuse, it matters most where positives are thinnest, and
it costs a script we needed anyway. Retiring a mis-specified test and
loosening a threshold until it passes are different acts, but they can look
alike from outside, so the whole episode is recorded here rather than dropped.

One thing the retired check did establish, and it was worth having: the R fold
file lines up with the full-population matrix row for row, all 46,010 of them.
That had been assumed since February and never verified.

### Fold counts

Ten folds throughout, on both matrices.

The rule set in advance was ten by default, dropping to five for any task
where ten would leave a (period, fold) cell with fewer than five positive
cases. The thinnest cell in the whole design is repeat ED use on the autism
subset, where the smallest period holds 70 positives, giving 7 per fold. That
clears the floor, so ten stands.

It clears it without much room, and the consequence is stated here rather than
discovered later: per-fold AUC for that task rests on roughly seven positive
cases, so its estimates will be noticeably noisier than the rest and its
confidence intervals wider. No headline should rest on that task's p-values.

### Household grouping

`hhid` is unique per row on the autism-subset matrix, all 6,088 of them, with
none spanning periods. The survey is cross-sectional, so this is expected.
Household grouping in the fold draw therefore reduces to row grouping and is
recorded as a deliberate no-op rather than an omission.

## Equal training-set sizes, which is required rather than optional

The three training sources are not the same size, and on the autism-subset
matrix the imbalance is severe and varies across periods:

| Test period | Same trains on about | Other trains on about | Ratio |
| --- | ---: | ---: | ---: |
| 2016-17 | 934 | 5,050 | 5.4 |
| 2018-19 | 919 | 5,067 | 5.5 |
| 2020-21 | 1,505 | 4,416 | 2.9 |
| 2022-23 | 2,121 | 3,731 | 1.8 |

Other is always the larger training set, by a factor that itself varies
threefold across periods. A period-to-period pattern in Other minus Same is
therefore exactly what training-set size alone would produce, with the early
periods most inflated. Read naively, that contrast would say the earliest
periods transfer differently, when what is different about them is that they
are small.

The full-population matrix has the same problem in milder form, 18,202 against
27,808, and there the direction of the size advantage flips between the two
test years, which is enough on its own to flip the sign of the contrast.

SOAK already solves this. Training on equal-sized subsets is what the `sizes`
mechanism is for; `sizes=0` trains every source at the size of the smallest,
it is implemented in our splitter, its stratification defect was fixed in
PR#9, and there are tests behind it.

**Therefore:** `sizes=0` is run in this pass on both matrices, and **Other
minus Same is reported only from the equal-size comparison.** The full-size
version of that contrast is not reported at all, because it is not
interpretable and reporting it beside the corrected one invites the reader to
average two numbers that do not mean the same thing.

All minus Same is reported at full size, because there the larger training set
is the treatment rather than a confound: the question is precisely whether
more data from other periods helps.

The splitter's rule is that every source larger than Same is also trained at
Same's size, which adds downsampled splits to each (subset, fold). The exact
count follows from the splitter and is recorded in each run's output rather
than asserted here; it will be checked against expectation before the runs go
out.

## Model, metrics, and tests

**Learner.** The validated lasso configuration, unchanged: logistic regression
with an L1 penalty, liblinear, intercept scaling 100, a 60-point penalty grid
from 1e-4 to 1, five-fold inner cross-validation on log loss, and
lambda-1se-analogue selection, as implemented in
`analyses/run_glmnet_replication.py`.

The argument for not changing it is that we validated one configuration
against R, and a different configuration here would forfeit that. The argument
against is that a configuration tuned for a 3% outcome is not obviously right
for outcomes running to 64%. I am keeping it, and recording that the choice
was made for comparability rather than because it is optimal for these
outcomes. If a reviewer asks whether the learner was tuned per outcome, the
answer is no, deliberately.

**Metrics.** Four, and the reason for each.

*AUC* is primary, because it is insensitive to prevalence and the outcomes
here range from 5.3% to 64.5%.

*Accuracy and percent error* are reported because percent error is the SOAK
paper's own metric. A caveat that applies to the replication does not carry
over: at the autism outcome's 3% prevalence, accuracy ran from 0.973 to 0.980
on every split and could not see what AUC saw. These outcomes are common
enough that percent error is live, and the old warning should not be imported
by habit. Prevalence is stated beside every accuracy-family number.

*Brier score and calibration slope* are added because AUC is invariant to any
monotone transformation of the predicted probabilities. A model can rank
children perfectly while systematically over- or under-stating how many lack
care. The prevalences above move from 9.1% to 14.1% across periods, so a model
trained on early periods and applied to 2020-21 may rank well and be badly
calibrated. Since the question is whether periods can be pooled, and the
manuscript downstream cares about levels rather than only ordering,
calibration is the property most likely to distinguish the training
strategies. The calibration slope is from a logistic regression of the outcome
on the logit of the predicted probability in the test set, where one is
perfect. Both come from predictions we already save, so they cost nothing.

**Confirmatory and exploratory.** Seven tasks with two contrasts across two or
four subsets is roughly forty tests before the sensitivity variant. At the
conventional threshold, two or three significant results are expected from
noise alone. Rather than correct, which the small number of folds makes
awkward, the plan designates in advance what it is confirming.

*Confirmatory, one per task:* All minus Same, pooled across subsets. Paired
differences are formed per (subset, fold) and a single one-sample t-test is
run over all of them, twenty differences on the full-population matrix and
forty on the autism subset. This is the comparison the manuscript relies on
and it is the only claim that will be stated as a finding.

*Exploratory, everything else:* per-subset contrasts, Other minus Same at
equal size, and all secondary metrics. Reported in full, described as
exploratory wherever they appear, and not the basis of any claim.

**Unit of analysis, and a limitation.** Within a (subset, fold), all three
training sources are scored on the same held-out children and move together,
so the fold is the unit rather than the split.

That is right about the test sets and only partly right about the training
sets, which overlap heavily from fold to fold. Cross-validation fold estimates
are positively correlated, so a paired t-test across folds underestimates
variance and rejects more often than its nominal rate. This is a known
property of the design rather than a defect in our implementation, and the
same test is used in the published analysis, so it is kept for comparability.
The consequence is that p-values near the threshold are more fragile than they
appear. The replication demonstrated this empirically: on one of four
contrasts the R and Python implementations returned opposite verdicts from
nearly identical estimates.

**No predictor-level claims.** These runs do not support statements about
which characteristics predict an outcome. `equivalence-margin.md` records that
two runs differing only in a random seed keep or drop five to eight different
features, with individual coefficients differing by up to 0.377. With 364
largely correlated one-hot columns that is expected behaviour for a lasso,
which selects one member of a correlated group more or less arbitrarily. Any
list of "top predictors" from a single fit would not survive rerunning with a
different seed, and we have our own evidence of that. See the follow-ups.

## Cross-matrix consistency, required before results are combined

The two matrices were built by different code at different times and encode
the same outcomes under different column names. Children with autism surveyed
in 2019 and 2020 appear in both.

Nothing has yet checked that they agree. Before any result from one is
presented alongside a result from the other, the overlapping years will be
compared: the share of children with autism reporting foregone care and each
ED definition, computed from each matrix over 2019 and 2020, should agree
within sampling error. A material disagreement means one of the two encodes
something differently from what its column name says, and everything resting
on that matrix would need rechecking.

## What this pass deliberately excludes

Recorded so that each is a choice rather than an oversight.

- **Survey weights.** These runs are unweighted and sample-level, stated in
  every caption. The weighted analysis is a separate deliverable and it is
  what the outcomes manuscript actually needs.
- **Autism diagnosis as a predictor** in the full-population tasks.
- **Per-outcome learner tuning.**
- **2024 data**, blocked on a file not yet obtained.
- **The full population across all survey years**, which needs a matrix that
  does not exist.

## Known limitations

- Every result is unweighted and describes the children surveyed, not the
  national population.
- Every association is cross-sectional. Nothing here identifies a cause.
- Fold-level tests are anti-conservative for the reason given above.
- Repeat ED use on the autism subset rests on about seven positive cases per
  test cell.
- Missing-data encoding is unverified, and until it is, these results are
  provisional.
- The learner was validated for a 3% outcome and is being applied unchanged to
  outcomes as common as 64%.

## Pre-registered follow-ups

Planned now so that running them later is not a reaction to results.

**Stability selection.** If the manuscript wants to name predictors, the
answer is not a single lasso fit. Refitting over resamples and reporting
selection frequency gives something that survives a change of seed, and it
yields a question worth asking directly: which predictors are selected in
every period against only some. That speaks to whether the structure of unmet
need has changed over time, which is closer to the real question than any AUC
contrast.

**Survey-weighted estimation.** Design-based analysis with strata, primary
sampling units and child weights, reported with odds ratios and confidence
intervals. This is what turns statements about the sample into statements
about the population.

**The outcomes manuscript's own analysis plan**, which this document is not
and does not substitute for.

## Amendments

Additions go below this line, dated, without altering anything above.

### 12 August 2026: "equal size" means within about half a percent

The section above says the Other-minus-Same contrast is reported from the
equal-size comparison. Having now inspected what the splitter actually
produces, the training sets it equalises are close but not identical, and the
plan should say so before any result exists rather than after.

The downsample keeps a proportional prefix within each (subset, outcome)
stratum, taking `floor(stratum_size * target / nominal_own)` rows from each.
Every stratum therefore rounds down, and the shortfalls accumulate: a source
spanning eight strata can finish up to eight rows below target. The target is
also the *nominal* size rather than any particular split's actual size, which
adds a row or two of wobble on top.

Measured on the autism-subset matrix, for the ED-any task:

| Period | Same, actual | Other at equal size | All at equal size | Largest gap |
| --- | ---: | ---: | ---: | ---: |
| 2016-17 | 933 to 935 | 930 to 934 | 929 to 931 | 0.6% |
| 2018-19 | 918 to 920 | 915 to 916 | 915 | 0.5% |
| 2020-21 | 1,504 to 1,505 | 1,501 to 1,503 | 1,499 to 1,501 | 0.4% |
| 2022-23 | 2,121 to 2,122 | 2,117 to 2,121 | 2,118 to 2,119 | 0.2% |

So "equal size" should be read as equal to within one percent, and the largest
discrepancy observed is six rows in nine hundred. Set against the 1.8 to 5.4
fold imbalance it corrects, this is immaterial, and it is a property of the
SOAK downsampling rule rather than a choice available to us. But a reader
comparing 935 against 929 is entitled to an explanation, and it should be here
rather than improvised later.

This touches only the exploratory contrasts. The confirmatory contrast,
All minus Same, is computed at full size and is unaffected.

### 24 August 2026: the leak audit was defective, and what replaces it

The first results on the autism-subset matrix came back with foregone care
predicted at AUC 0.87 against emergency department use at 0.71. Unmet care
need should be the harder problem, not the easier one, and that gap is what
prompted this.

**What was wrong.** The leak audit recorded above examined four variables:
`k4q20r`, `k4q02_r`, `k4q22_r` and `k4q24_r`. Every one was found by matching
column names against the outcome's survey code, `k4q27`. That method cannot
find a variable whose name shares no characters with the outcome's, and it
cannot say what it missed. It is not that the audit was applied carelessly;
the instrument was wrong for the job.

**What it missed.** `c4q04`, "Frustrated In Efforts to Get Service". In the
fitted models it carries a mean coefficient of -1.31 at the `Never` level and
is selected in 120 of 120 splits, against a next-largest coefficient of 0.15.
Among 2,738 children whose families were never frustrated, 28 reported
foregone care, a rate of 1%, rising to 55% among those always frustrated. That
one variable alone scores AUC 0.8347, against 0.8709 for the full 292-feature
model. The other 291 features contribute 0.036 of AUC between them.

The pre-registered `foregone_care_strict` variant did not catch this. It
removed `k4q22_r` and `k4q24_r` and moved AUC only from 0.8709 to 0.8591, so
that sensitivity question is answered: those two columns are not the problem
and they stay. The variant did its job; it was simply aimed at the wrong
variables, for the same reason the audit was.

**What is not wrong.** The follow-up items asked only of families who answered
yes to `k4q27` are absent from both matrices. `issuecost`, `notopen`,
`transportcc`, `appointment`, `available`, `notelig`, `treatneed`, `k4q26` and
the entire `k4q28x` family were checked by name and none is present. The
matrices were built without the logical-skip follow-ups, which is a real point
in their favour and worth recording as a finding rather than as an absence of
bad news.

**The replacement.** `analyses/audit_feature_constructs.py` joins every
feature column to the survey's own `label var` text and groups the features by
what their questions ask. Both matrices now resolve at 100%, up from 55% on
the full-population matrix before a punctuation-matching defect was fixed. The
audit reads several years' `.do` files at once, because a harmonized matrix
carries names drawn from different years and no single year covers it. Its
output lives in `analyses/feature-audit/`.

The rule this plan now adopts:

> Exclude features whose question concerns the process of obtaining care in
> the same twelve-month window as the outcome. Keep features describing the
> child's or family's circumstances. Every exclusion cites a label from the
> feature audit.

**The rule is post-hoc and that has to be stated.** It was written after
seeing that `c4q04` had a large coefficient. Its defence is that it is stated
in terms of construct rather than of any variable or any coefficient size, it
would have excluded `c4q04` at any magnitude, and it is applied uniformly to
all four outcomes including ones whose features have not been examined. That
is a weaker position than a rule fixed in advance, and it is recorded here as
such rather than presented as though it had been.

**Exclusions under the rule.**

| Matrix | Excluded | Label |
| --- | --- | --- |
| both | `c4q04` | Frustrated In Efforts to Get Service |
| both | `k5q10` | Need a Referral |
| autism subset | `k5q11` | Need a Referral - Problem |
| autism subset | `k5q20_r` | Arrange Or Coordinate Care Among Doctors |
| autism subset | `k5q21` | Arrange Or Coordinate Care Extra Help |
| full population | `k3q20` | Health Insurance - Benefits Cover Services |
| full population | `k3q22` | Health Insurance - Allow to See Provider |
| full population | `menbevcov` | Health Insurance - Cover Mental Behavioral Needs |

These are added as `_conservative` outcome variants rather than replacing the
primary specification. All specifications are reported. Amending the existing
`foregone_care_strict` definition after seeing which variable was large would
have been fitting the specification to the answer.

**One exclusion the rule does not decide: insurance adequacy.** `k3q20`,
`k3q22` and `menbevcov` ask how often insurance covered services, allowed the
chosen provider, or covered mental and behavioural needs. These can be read as
attributes of the plan, which is a circumstance, or as experiences of seeking
care during the year, which is an episode. This plan reads them as episodes,
because each asks about a frequency of encounters over the past twelve months
rather than about what the policy says. A reasonable analyst could decide the
other way. Recorded as a judgment, made once, applied consistently.

Insurance type, coverage gaps and the specific coverage-source items stay:
those describe what the family has, not what happened when they sought care.

**What this does not settle.** These exclusions are chosen from the tier the
audit flags plus a reading of the tier below it. That is far better than name
matching, but it is still a human reading of 293 labels and it may miss
something. The audit output is committed so that the reading can be checked
and redone.

### 24 August 2026: label drift on k4q20r, examined and dismissed

The feature audit compares labels across survey years and flags any stem whose
wording changed. `k4q20r`, the variable this plan already records a correction
about, is one of them.

2016 labels it "Doctor Visit - How Many Times". Every year from 2017 to 2024
labels it "Preventive Visit - How Many Times". If the 2016 wording described a
different question, then for that year zero visits would imply no emergency
visits, and the original leak concern this plan overturned would have been
correct for part of the data.

Two pieces of evidence say the wording changed and the question did not. The
response options are identical in every year checked: 0 visits, 1 visit, 2 or
more visits. And the distribution is flat across all four periods, at
10.8/52.4/36.8, 15.0/51.7/33.3, 14.1/54.3/31.6 and 12.3/53.9/33.8. A question
that counted all doctor visits in 2016 and preventive visits afterwards would
put noticeably more of the 2016-17 period in the top band, since a well-child
check plus any sick visit would move a child up. It has slightly fewer. The
2016-17 period pools 2016 with 2017, which would halve any 2016-only shift,
but halved from an effect that large would still be visible.

`k4q20r` stays. The plan's entry above stands, and its label citation should
be read as 2017 onward rather than as 2024, which is the file it was
originally checked against.

The general lesson, since this variable has now turned on a single-file label
twice: a label citation should name the survey year it came from. The audit
output supports that and future entries should do it.
