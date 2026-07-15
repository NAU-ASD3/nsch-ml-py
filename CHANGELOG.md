# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow a date-based scheme: `YYYY.M.DD`.

## [Unreleased]

## [2026.7.8] - 2026-07-08

### Added

- `nsch_ml.soak` (PR #2): stratified fold assignment with a precomputed passthrough for reproducing the R analysis's folds, the same/other/all split iterator with the `sizes=0` stratified downsampling, and the inner ignore-group k-fold. Semantics pinned against the mlr3resampling source (archived `ResamplingSameOtherCV` 2024.9.6; current `ResamplingSameOtherSizesCV`).

## [2026.7.7] - 2026-07-07

### Added

- Initial repository scaffolding (PR #1):
  - `pyproject.toml` with `[project]` metadata, the analysis dependency stack, and configuration for ruff, mypy, pytest, and coverage.
  - GitHub Actions CI workflow with lint, matrix test (Python 3.11–3.13 on Linux; 3.13 on macOS and Windows; a lowest-direct floor check), build, docs, and dependency-review jobs.
  - GitHub Actions docs workflow publishing the mkdocs site to GitHub Pages.
  - Pre-commit hooks mirroring CI checks, Dependabot configuration, issue and PR templates, and `CODEOWNERS`.
  - Documentation skeleton (`mkdocs.yml`, `docs/index.md`, `docs/design-decisions.md`).
  - Planning documents under `planning/`: the development plan and the Workstream C scoping document.
  - The `src/nsch_ml/` package with a smoke test.
