# nsch-ml

Python replication of the ASD3 Outcomes Project's published NSCH machine-learning analysis: SOAK cross-validation across survey periods, penalized logistic models, XGBoost with SHAP explanations, SHAP-based clustering, and fairness analyses. The package consumes the harmonized dataset produced by [nsch-py](https://github.com/NAU-ASD3/nsch-py) and validates its results against the original R outputs on held-out predictions.

The package is under active development; modules land PR by PR. Until the API reference exists, the best orientation is:

- [Design decisions](design-decisions.md): why the package is built the way it is.
- The planning documents in [`planning/`](https://github.com/NAU-ASD3/nsch-ml-py/tree/main/planning) at the repo root: the full analysis scope and the build sequence.
- [`CONTRIBUTING.md`](https://github.com/NAU-ASD3/nsch-ml-py/blob/main/CONTRIBUTING.md): development setup and conventions.
