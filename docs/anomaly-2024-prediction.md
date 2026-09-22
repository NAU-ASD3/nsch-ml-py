# A prediction about the 2024 data

Chris Reger, 26 August 2026.

This is written before the 2024 survey file is in hand. Its whole value is the
timestamp: it commits to a prediction and to how that prediction will be tested,
so the test is a test rather than a description.

Like `equivalence-margin.md`, this is append-only. Amendments go at the bottom,
dated, without altering the text above.

## What we observed

Predicting repeat emergency department use, meaning two or more visits in the
past year, among children with autism, a model trained on earlier periods does
markedly worse at the 2022-23 period than a model trained on 2022-23 itself.
At matched training size, and pooling across four periods, the contrast is
about -0.03 AUC.

Broken down by period, at equal training size, across five runs varying the
fold draw, the feature specification and the fold count:

| Period | Range across runs | Significant |
| --- | --- | --- |
| 2016-17 | -0.037 to -0.049 | no |
| 2018-19 | -0.042 to -0.058 | no |
| 2020-21 | -0.005 to +0.029 | no |
| **2022-23** | **-0.041 to -0.058** | **yes, every run** |

The same outcome on the full-population data, where cases are ten times more
plentiful, shows nothing comparable. The broader any-visit definition of ED use
shows nothing at 2022-23 either.

## Why this is not yet a finding

We found it by inspecting eight per-period contrasts with no prior hypothesis,
then re-ran the most extreme one five times. Re-running with different fold
draws tests whether an estimate is stable. It does not test whether that cell
deserved to be singled out. Take the most extreme of eight cells, repeat it, and
you get consistent extreme values whether or not anything is there.

Everything above is exploratory. The 2024 data is the only genuine
out-of-sample test available to us, and it is only a test if the prediction is
written first.

## What we think is happening

The leading explanation, reached by elimination rather than by evidence.

Thinness is ruled out. Halving the fold count doubled the cases per test set
and the effect held; three fold draws gave a stable estimate; the same outcome
with ten times the data shows nothing.

Label drift is ruled out. Twenty-five features have labels that change between
survey years, and twenty-three of them change at 2017 in what appears to be a
single relabelling exercise. None change at 2022.

A pandemic explanation is not supported. 2020-21 is the period that transfers
best. The discontinuity is at 2022, not 2020.

What remains is the outcome's own coding. `hospitaler` carried three response
levels through 2021 and four from 2022, splitting the top band. Both our ED
definitions are built from the two stable bottom levels, and the share of
children with two or more visits does not jump at 2022, so a simple version of
this is already argued against.

But the asymmetry fits. Splitting the top band changes how respondents
distribute themselves *within* the two-or-more group, which is exactly where
`ed_repeat` lives and where `ed_any` does not. That `ed_any` shows nothing while
`ed_repeat` does is the one piece of positive evidence for this explanation.

## The prediction

**2024 is on the four-level coding, the same as 2022-23.** If the coding change
is responsible, 2024 belongs to the same regime as 2022-23 and a different one
from everything earlier.

**Primary prediction.** When the survey years are grouped into two subsets,
2016 to 2021 against 2022 to 2024, the equal-size Other-minus-Same contrast for
repeat ED use among children with autism will be negative, with a point
estimate between -0.02 and -0.08.

**Secondary prediction.** With 2024 as a fifth two-year period in its own right,
the equal-size Other-minus-Same contrast at the 2022-23 period will remain
negative and in the range already observed, -0.02 to -0.08.

**Control prediction.** Neither contrast will be significantly negative for
ED use at one or more visits, run identically on the same data.

## What would falsify this

**The anomaly was noise.** Adding 2024 moves the 2022-23 contrast to within
0.01 of zero, or changes its sign. In that case the five confirming runs were
five looks at one lucky draw, and this document is the record of us being wrong
in public.

**The regime explanation is wrong but something is there.** The two-subset
contrast comes back near zero while the 2022-23 period contrast stays negative.
That would mean 2024 does not group with 2022-23, so whatever happened was
specific to those two years rather than a change persisting forward.

**It is not about the coding.** The control fails, meaning ED use at one or
more visits shows the same pattern. The coding change touches only the top
band, so a matching effect in the broader definition would point elsewhere.

## How this will be run, decided now

1. The 2024 data is characterised first: row counts, outcome prevalence, and a
   crosswalk audit, exactly as any new year would be. That work does not touch
   the contrasts below.
2. Folds are drawn by `draw_outcome_folds.py`, stratified on (subset, outcome),
   ten folds, seed 1, as the plan specifies.
3. The two runs above are executed once each, with `sizes=0`, using the
   existing runner and no changes to the learner.
4. The contrasts are read once, from `extension_contrasts.py`, and recorded
   here as an amendment whatever they show.

**No second look.** If the result disappoints, we do not vary the period
grouping, the fold count or the seed and try again. Any such run goes in
`analyses/results/variants/` and is exploratory, and it does not replace the
result recorded under this protocol.

**2024's period assignment is fixed here** because it changes the test: 2024
forms its own subset rather than being folded into 2022-23. That decision is
made now, before the data, and for this reason.

## What this cannot establish

A confirmed prediction supports the regime explanation. It does not distinguish
a change in the survey instrument from a real change in how children with
autism used emergency care from 2022 onward. Both predict the same pattern.
Separating them would need something outside this data, and we should say so
rather than let a confirmed prediction imply more than it shows.

## Amendments

None yet. Results go below this line, dated, without altering anything above.
