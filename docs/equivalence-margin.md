# What counts as a successful port

Chris Reger, 6 August 2026.

This fixes the standard before the comparison is run. Nothing in here has
been checked against a Python result; the only numbers are from R runs
compared against each other. The git timestamp is the point. Everything we
tried previously was judged after we could see how it did, which is exactly
the practice the equivalence note argues against, and the only way out of
that is to write the target down first.

Toby, Ben and Olivia have not seen this yet. It goes to them Wednesday. If
they want the thresholds moved, moving them is a one-line edit to a document
that already exists, which beats arriving with a blank page.

## The reference is not a fixed point

Three R runs exist that differ in one thing: the `cv.glmnet` inner
cross-validation seed. Same machine, same afternoon, same package build,
same outer folds. Whatever they disagree about is noise in the reference,
and nothing we write in Python can be closer to one of them than they are to
each other.

| | seed1 vs seed2 | seed1 vs seed3 | seed2 vs seed3 |
|---|---|---|---|
| Coefficient mean absolute difference | 0.00252 | 0.00430 | 0.00391 |
| Coefficient max absolute difference | 0.224 | 0.377 | 0.292 |
| Features where selection differs | 1.4% | 2.3% | 2.0% |
| Held-out ranking, worst Spearman | 0.9654 | 0.9678 | 0.9678 |
| Held-out ranking, mean Spearman | 0.9962 | 0.9920 | 0.9946 |
| Fold-level AUC, mean absolute difference | 0.00028 | 0.00059 | 0.00050 |

Two things in that table are worth sitting with.

The lasso is not stable. Largest coefficients run to about 1.04, and two
seeds can differ by 0.377 on a single weight. Roughly five to eight features
per fit are kept by one seed and dropped by the other. With 364 largely
correlated one-hot columns that is unsurprising — lasso picks one member of a
correlated group more or less arbitrarily — but it does mean coefficient
agreement cannot be a pass/fail line. The reference does not have it.

Neither are the held-out rankings. On the worst split two seeds disagree on
about three and a half percent of pairs. That is a property of the analysis,
not of anything we have written.

The fold-level AUC, the crudest of the three, is the one thing that holds
still. Averaging over 1,819 to 2,781 children washes out the instability
underneath. That is a point in favour of Toby's choice to report AUC, and it
is worth telling him: the finer the quantity, the less reproducible SOAK is
against its own seed.

## The margins

| Quantity | Reference spread | Margin | Gated |
|---|---|---|---|
| Spearman of held-out scores, per split | 0.965 floor | **at least 0.95** | yes |
| Fold-level AUC, mean absolute difference | 0.00059 | **at most 0.002** | yes |
| Probability-scale MAD | not yet measured | **at most 0.01**, provisional | yes, once available |
| Coefficient mean absolute difference | 0.0043 | none | reported only |
| Feature selection agreement | 97.7% | none | reported only |

The port passes if it clears every gated line on every split. Failing one is
a failure, not an invitation to look for a fourth measure that passes.

**Spearman at 0.95.** Sits just under the reference's own worst pair. If the
port ranks held-out children as consistently with R as two R seeds rank them
with each other, that is as good as this analysis gets.

**AUC at 0.002.** About three times the reference spread. The multiplier is a
judgement call and I am not going to pretend otherwise. It is loose enough to
absorb a different penalty-selection procedure and tight enough to sit well
under the 0.0073 that separates two package builds, which is the scale at
which conclusions actually move.

**Probability MAD at 0.01, provisionally.** The intercept is not stored in
R's saved learners, so this cannot be computed until a rerun saves
predictions. On the probability scale, with 3% prevalence, 0.01 is not
obviously the right number and I would rather mark it provisional than
pretend it was derived. It gets revisited once we can see the R-internal
spread on the same quantity, using the same seed comparison as everything
else here.

**Coefficients and selection are reported, not gated.** Gating on a quantity
where the reference disagrees with itself by a third of its largest weight
would be measuring the lasso's instability and calling it our error.

## About the 99% classification agreement

The June plan proposed 99% agreement on the binary decision. That was a
sensible instinct written before anyone had looked at the data, and it does
not survive contact with this outcome. Prevalence is about 3%, so a model
that says "no" to every child agrees with any other model 97% of the time.
Ninety-nine percent is close enough to that floor to be nearly free.

It stays as a reported number because it was proposed and because it is easy
to read, but with the prevalence stated next to it every time. It is not
gated.

## What this does not replace

The paired t-test on 9 degrees of freedom, per test subset, reported with the
two figures after the SOAK paper's Figures 4 and 6 to 7, is the analysis. It
answers whether training on all years beats training on one, which is the
question SOAK exists to ask. It is computed for both implementations, and it
stays in `docs/replication-equivalence.md` and in `analyses/soak_ttests.py`.

It is not the equivalence criterion, for two reasons set out in that
document. Across ten R runs the verdicts disagree on two of four contrasts,
so the reference does not meet the standard against itself. And paired tests
on overlapping cross-validation folds understate variance (Dietterich 1998;
Nadeau and Bengio 2003; Bengio and Grandvalet 2004), so the p-values are
optimistic on both sides.

Using each tool for the question it answers is not a rejection of the method.
The t-tests tell us whether pooling helps. The margins above tell us whether
our code reproduces the study. Asking one to do the other is what sent the
previous week in circles.

## Conditions

Fixed before any comparison:

- Reference is `NSCH_seed1`, on `mlr3learners` 0.14.0.9000. The build has to
  be recorded because it moves results by 0.0073, further than anything else
  we have measured.
- Comparison is against our **lasso** fits, in
  `analyses/glmnet_replication_lasso.csv`. R's `cv_glmnet` runs alpha = 1,
  confirmed from the saved coefficients: 65 nonzero of 364. Comparing our
  ridge against R's lasso would test a choice we made rather than the port.
- Sixty splits, but only thirty distinct fitted models. For any fold, "same"
  on 2019 and "other" on 2020 train on identical rows, and their coefficients
  are bit-identical. Coefficient statistics run over the thirty.
- glmnet's coefficients point toward `y = 0` in this fixture. Every split
  reproduces exactly `1 - AUC` before the sign is corrected. Anything
  comparing coefficients has to account for that or it will report 364 wrong
  signs per split.
- All 364 features, including `Selected_Child_Weight` and
  `Unique_Household_ID`, both of which R's lasso drops in all 60 splits.

## If it fails

Then it fails, and the interesting work starts. A margin set in advance and
missed tells us something; a margin adjusted afterwards tells us nothing.

The one revision I will accept without embarrassment is the provisional
probability-scale figure, because it was set without a reference measurement
and is labelled that way here.
