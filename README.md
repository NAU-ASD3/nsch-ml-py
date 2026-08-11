# nsch-ml

A Python replication of the ASD3 Outcomes Project's published NSCH
machine-learning analysis: SOAK cross-validation across survey periods,
penalized logistic models, boosted trees, and fairness analyses.

**Status: pre-alpha.** The repository is scaffolded and modules land PR by
PR. The `cv_glmnet` port and its validation are complete; the other four
learners are not started.

## Why this package

The original analysis is written in R, spread across
[`vas235/ASD3-machine-learning`](https://github.com/vas235/ASD3-machine-learning)
and its data-prep repository, and built on `mlr3resampling`'s SOAK
implementation.

This package reproduces that analysis in Python. Reproducing it is not the
goal in itself: the project needs to extend the work to 2024 data, to three
new outcomes, and to a survey-weighted design, and none of that is
interpretable unless the Python code first gives the same answers as the R
code on the data already published.

The port is validated on held-out predictions rather than on fitted
coefficients. That is the equivalence the two ecosystems can share. glmnet
and scikit-learn standardize their inputs differently and express penalty
strength differently, so their coefficients are not directly comparable even
when both are correct. What can be compared is what the models predict for
the same children.

[`docs/replication-equivalence.md`](docs/replication-equivalence.md) reports
how the two implementations compare and what the port can and cannot be used
for. It assumes no background on the original study.

The harmonized multi-year NSCH dataset comes from
[`nsch-py`](https://github.com/NAU-ASD3/nsch-py). This package does no
harmonization of its own, a boundary explained in
[`docs/design-decisions.md`](docs/design-decisions.md).

## The repo family

- [`nsch`](https://github.com/NAU-ASD3/nsch): the R harmonization package.
- [`nsch-py`](https://github.com/NAU-ASD3/nsch-py): its Python port, which
  produces the dataset this package analyzes.
- [`nsch-ml-paper`](https://github.com/NAU-ASD3/nsch-ml-paper): the paper
  repository.
- This repository: the analysis replication and, later, the survey-weighted
  extension.

## Documentation

Published at <https://nau-asd3.github.io/nsch-ml-py/>.

- [Replication equivalence](docs/replication-equivalence.md): how the port
  compares to the R analysis, and what follows from that.
- [Equivalence margin](docs/equivalence-margin.md): the pass/fail standard,
  fixed in writing before the comparison was run.
- [Design decisions](docs/design-decisions.md): why the package is built the
  way it is.

An API reference will follow as the package gains functions.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, conventions, and the PR
workflow.

## Citation

If you use this package in research, please cite the SOAK paper (Hocking et
al., 2026, _Statistical Analysis and Data Mining_,
[doi:10.1002/sam.70055](https://doi.org/10.1002/sam.70055)) and the
underlying NSCH data per the
[Census Bureau guidelines](https://www.census.gov/programs-surveys/nsch.html),
in addition to this package.

## License

MIT. See [`LICENSE`](LICENSE).

## Acknowledgements

This work is supported by the NIH Autism Data Science Initiative (ASD3
Outcomes Project) at Northern Arizona University.
