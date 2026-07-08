"""Python replication of the ASD3 NSCH machine-learning analysis.

SOAK cross-validation, penalized models, SHAP, and fairness analyses on
the harmonized NSCH dataset produced by nsch-py. See
planning/workstream-c-scoping.md for the full analysis scope and
planning/development-plan.md for the build sequence.
"""

from importlib.metadata import version

# Single source of truth for the version is pyproject.toml; reading it from
# the installed metadata avoids maintaining the number in two places.
__version__: str = version("nsch-ml")
