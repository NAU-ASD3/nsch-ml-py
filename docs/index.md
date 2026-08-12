# nsch-ml

A Python replication of the ASD3 Outcomes Project's published NSCH
machine-learning analysis: SOAK cross-validation across survey periods,
penalized logistic models, boosted trees, and fairness analyses.

The package reads the harmonized dataset produced by
[nsch-py](https://github.com/NAU-ASD3/nsch-py) and validates its results
against the original R outputs on held-out predictions.

Development is ongoing and modules land PR by PR. Until the API reference
exists, these are the places to start:

- **[Replication equivalence](replication-equivalence.md)** reports how
  closely the Python port matches the R analysis, what differs and why, and
  what the port can be used for. It assumes no familiarity with the original
  study, and it is the right first read.
- **[Equivalence margin](equivalence-margin.md)** is the pass/fail standard
  the comparison was judged against, committed before the comparison was
  written.
- **[Extension analysis plan](extension-analysis-plan.md)** does the same job
  for the work that follows the replication: the three service outcomes, the
  populations each is valid for, what is removed from the features and why,
  how folds are drawn, and what this pass deliberately leaves out. Committed
  before the first model was fitted against any of those outcomes.
- **[Design decisions](design-decisions.md)** records the choices a
  maintainer might otherwise re-litigate: why predictions and not
  coefficients, why we wrote our own splitter, what the label conventions
  are.
- **[`CONTRIBUTING.md`](https://github.com/NAU-ASD3/nsch-ml-py/blob/main/CONTRIBUTING.md)**
  covers setup, the checks a change has to pass, and the PR workflow.
