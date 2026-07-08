# nsch-ml

Python replication of the ASD3 Outcomes Project's published NSCH machine-learning analysis: SOAK cross-validation across survey periods, penalized logistic models, XGBoost with SHAP explanations, SHAP-based clustering, and fairness analyses.

**Status: pre-alpha.** The repository is scaffolded; modules land PR by PR per [`planning/development-plan.md`](planning/development-plan.md).

## Why this package

The original analysis lives in R across [`vas235/ASD3-machine-learning`](https://github.com/vas235/ASD3-machine-learning) and its prep repo, built on `mlr3resampling`'s SOAK implementation. This package reproduces it in Python and validates the port against the cached R outputs on held-out predictions, which is the equivalence the two ecosystems can actually share; coefficient-level agreement is unreachable because glmnet and scikit-learn standardize and parameterize their penalties differently. The validated replication then serves as the base for the project's survey-weighted extension. The full analysis scope, the validation criteria, and the extension are laid out in [`planning/workstream-c-scoping.md`](planning/workstream-c-scoping.md).

This package consumes the harmonized multi-year NSCH dataset produced by [`nsch-py`](https://github.com/NAU-ASD3/nsch-py). It does no harmonization of its own; that boundary is deliberate and documented in [`docs/design-decisions.md`](docs/design-decisions.md).

## The repo family

- [`nsch`](https://github.com/NAU-ASD3/nsch): the R harmonization package.
- [`nsch-py`](https://github.com/NAU-ASD3/nsch-py): its Python port; produces the dataset this package analyzes.
- [`nsch-ml-paper`](https://github.com/NAU-ASD3/nsch-ml-paper): the paper repo.
- This repo: the analysis replication and, later, the survey-weighted extension.

## Documentation

Documentation is published at <https://nau-asd3.github.io/nsch-ml-py/>: the design decisions behind the package now, with an API reference to follow as the package gains functions. Working planning documents live in [`planning/`](planning/), outside the published site.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, test conventions, and the PR workflow.

## Citation

If you use this package in research, please cite the SOAK paper (Hocking et al., 2026, Statistical Analysis and Data Mining, [doi:10.1002/sam.70055](https://doi.org/10.1002/sam.70055)) and the underlying NSCH data per the [Census Bureau guidelines](https://www.census.gov/programs-surveys/nsch.html), in addition to this package.

## License

MIT. See [`LICENSE`](LICENSE).

## Acknowledgements

This work is supported by the NIH Autism Data Science Initiative (ASD3 Outcomes Project) at Northern Arizona University.
