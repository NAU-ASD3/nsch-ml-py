# nsch-ml-py development plan

Chris Reger · ASD3 Outcomes · July 2026

This is the working plan for standing up `nsch-ml-py` and sequencing its first PRs. The full analysis scope, the validation criteria, and the survey-weighted extension live in [`planning/workstream-c-scoping.md`](workstream-c-scoping.md); this document is only about how the repo gets built. When the two disagree, the scoping doc wins.

## What this repo is

The Python replication of the published 2016–2023 ASD SOAK analysis, validated against the R outputs on held-out predictions, plus (later) the survey-weighted extension. It consumes the harmonized dataset that [`nsch-py`](https://github.com/NAU-ASD3/nsch-py) produces. It does not redo any harmonization.

## Conventions

Carried from `nsch-py` unchanged: uv for environments and packaging, hatchling build backend, src layout, ruff, mypy strict, pytest with coverage, mkdocs-material published to GitHub Pages, pre-commit mirroring CI, Dependabot with weekly grouped updates, the issue and PR templates, date-based versions (`2026.M.DD`) with a Keep-a-Changelog `CHANGELOG.md` entry per PR, stacked PRs in dependency order with the ⚠️ stacked-PR header, squash-and-merge only, and the test style: plain asserts, full-vector comparisons, synthetic data, one behavior per test. The local gate before any review request is the same too:

```bash
uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy src/
```

Deliberate differences from `nsch-py`, so nobody has to guess later:

1. **Dependencies.** Runtime: `polars`, `scikit-learn` (models and the clustering), `xgboost`, `shap`, `pandas` (the sklearn boundary). Not carried over: `pyreadstat`, `httpx` (harmonization concerns). Deferred: `svy` until the install question is settled, and `python-glmnet` as an optional cross-check extra rather than a hard dependency.
2. **No `release.yml`.** We're not publishing to PyPI (out of scope per the scoping doc), so no tag-triggered release workflow until that changes.
3. **`CODEOWNERS` added.** `nsch-py` doesn't have one; this repo does, with Ben and David on `*`, since they're the named reviewers.
4. **A top-level `planning/` directory.** Working documents (this file, the scoping doc) live there, outside `docs/`, so they don't publish to the documentation site. `docs/` stays user-facing: index, design-decisions, onboarding, API reference later.
5. **Fixture strategy.** The NSCH_autism validation fixture is 46,010 rows by 364 columns, too heavy to commit. Tests that need it download it from the Zenodo record (10.5281/zenodo.18273949) with a pinned checksum and cache it; those tests skip when offline. The one-line golden header from the R `2016_2023_SOAK.csv` is tiny and does get committed under `fixtures/`.

## Layout

```
src/nsch_ml/
  data/      prep: metro_yn imputation, missing-data drops, derived vars, period bins, conditional one-hot
  soak/      the splitter and the inner ignore-group CV; design-aware folds land here later
  learners/  thin wrappers: penalized logistic (ridge and lasso), xgboost, knn, rpart-equivalent, featureless
  shap/      TreeSHAP extraction, out-of-fold averaging, base-variable collapse
  cluster/   k-means over SHAP vectors, CH/DB scoring, the characterization
  fairness/  the fairness-subset driver
  report/    ROC and the high-specificity zoom, confusion matrices at the FNR targets, coefficient tables
tests/       same layout; plain asserts, synthetic data, full-vector
docs/        index, design-decisions, onboarding
planning/    this plan, the scoping doc
fixtures/    the committed golden header; Zenodo-fetched fixtures cache here (gitignored)
```

## PR sequence, Phase 1

Mirrors how `nsch-py` was built: a scaffold PR, then one focused PR per module, stacked in dependency order. Estimates are rough.

| PR | Adds | Depends on |
|---|---|---|
| #1 | Repo skeleton: pyproject, CI, pre-commit, Dependabot, templates, mkdocs skeleton, empty `src/nsch_ml` with a smoke test, `CLAUDE.md`, this plan and the scoping doc under `planning/` | — |
| #2 | `soak/`: fold assignment stratified on (subset, outcome), the same/other/all iterator, seeded downsampling, the inner ignore-group 5-fold | #1 |
| #3 | Splitter equivalence: the Zenodo fixture fetch, and the test asserting fold IDs and same/other/all index sets match the R assignments | #2 |
| #4 | `data/` prep up to the model matrix: metro_yn imputation, the >10% drop and complete-case, race7/povlev4/fairness copies, period bins, the age-at-diagnosis side file | #1 |
| #5 | `data/` conditional one-hot, written test-first against the committed golden header | #4 |
| #6 | `learners/`: penalized logistic (ridge and lasso), featureless, knn, rpart-equivalent; the C-to-lambda mapping documented in design-decisions | #1 |
| #7 | `report/` for Phase 1: AUC and accuracy per period and per same/other/all, the paired t-test comparison, coefficient tables | #2, #6 |
| #8 | The Monsoon driver: a job-array entry point wrapping the splitter, plus the run configs | #2–#7 |

Phase 2 (xgboost tuning, SHAP, the clustering, fairness) gets its own PR table once Phase 1 lands; the module homes already exist in the layout. The design-aware resampling for the extension goes into `soak/` after that, per the scoping doc.

The one-hot golden test in #5 is the highest-priority test in the repo. The scoping doc explains why: a wrong reference level shifts every model baseline silently. Pull the header rows of the R `2016_2023_SOAK.csv` and `2016_2023_SOAK_categories.csv` before writing the encoder, and pin the exact column set and per-variable reference level, not the count.

## Open items for this repo specifically

- [ ] Confirm the repo is public. The rest of the family is, and this plan plus the scoping doc will be visible.
- [ ] Settle the `svy` install question before it's needed (scoping doc §4 and §7).
- [ ] Decide whether `nsch` (the Python package) becomes a dependency or whether this repo only reads the cached harmonized dataset. Current lean: read the cached dataset, keep the dependency surface small, revisit if we need `get_clean_data()` live.
- [ ] Branch protection on `main`: require PR review and green CI. Set after PR #1 merges so the scaffold isn't blocked by its own rules.
