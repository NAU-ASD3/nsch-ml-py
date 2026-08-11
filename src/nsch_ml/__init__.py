"""Python replication of the ASD3 NSCH machine-learning analysis.

SOAK cross-validation, penalized models, boosted trees, and fairness
analyses on the harmonized NSCH dataset produced by nsch-py.

The port is validated on held-out predictions, not on fitted coefficients;
``docs/replication-equivalence.md`` reports how it compares to the R analysis
and ``docs/design-decisions.md`` explains why the package is built this way.
"""

from importlib.metadata import version

# Single source of truth for the version is pyproject.toml; reading it from
# the installed metadata avoids maintaining the number in two places.
__version__: str = version("nsch-ml")
