---
name: Implementation task
about: A scoped piece of the analysis pipeline to build
title: ""
labels: enhancement
assignees: ""
---

## What needs to happen

<!-- One or two sentences. What function or behavior should exist after this is done? -->

## Where it goes

- Module: `src/nsch_ml/<module>.py`
- Tests: `tests/test_<module>.py`

## Reference

<!--
Link whatever defines the correct behavior for this task. Depending on the
module, that's usually one of:
  - a script in vas235/ASD3-machine-learning (most analysis behavior)
  - NSCH_autism_reproduce.R in tdhock/cv-same-other-paper, or
    mlr3resampling itself (splitter and resampling behavior)
  - a committed golden fixture (the one-hot header)
  - the SOAK paper or the Census NSCH documentation (methods and design facts)
-->

- Reference implementation or source: <!-- link it -->
- Plan: `planning/development-plan.md` PR row and/or `planning/workstream-c-scoping.md` section: <!-- name it -->

## Acceptance criteria

- [ ] The behavior matches the R reference at the level the plan specifies (predictions, index sets, or exact values, as applicable).
- [ ] Tests are written first, use synthetic in-memory data, and assert on full columns.
- [ ] Randomness, if any, is controlled by an explicit seed parameter.
- [ ] A NumPy-style docstring is present, with a doctest example for anything non-trivial.
- [ ] `uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy src/` passes locally.
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]` referencing the PR.

## Notes

<!-- Anything that will save the next person time: a quirk of the original analysis, a glmnet-versus-sklearn difference, a seed detail. -->
