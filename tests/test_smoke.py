"""Smoke test: the package imports and reports a version.

Exists so that CI exercises the install itself from the very first PR.
pytest exits nonzero when it collects no tests, so this file also keeps
the local CI-mirror chain runnable before any real modules land.
"""

import nsch_ml


def test_package_imports_and_has_version() -> None:
    # Date-based scheme: YYYY.M.DD, so the year is always the first field.
    assert nsch_ml.__version__.startswith("2026.")
