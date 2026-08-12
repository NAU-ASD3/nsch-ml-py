"""The outcome definitions the extension analyses share.

Three things have to agree about an outcome: the fold draw stratifies on it,
the model fits it, and the leak audit decides which columns to remove because
they would give it away. If those three read the definition from three places,
they will eventually disagree, and the failure is silent: the run completes and
the numbers look plausible.

So each outcome is defined once, here, and every script imports it.

The positive-class rules follow docs/extension-analysis-plan.md:

  foregone care     the child needed health care in the past year and did not
                    receive it
  ED use, any       one or more emergency room visits, which is the complement
                    of the "no visits" indicator
  ED use, repeat    two or more visits. The survey records visits in three
                    bands and the matrix carries indicators for the lowest
                    two, so "two or more" is the rows where both are zero.

Only the outcome's own columns are dropped from the features. Neighbouring
columns that merely correlate with an outcome stay in; predicting an outcome
from its correlates is the analysis, not a leak. The one column that looked
like a leak on inspection, k4q20r, turned out to count preventive visits
rather than all visits, so it cannot imply anything about emergency use and it
stays.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

FOREGONE_CARE_COLUMN = "Needed_Health_Care_Not_Received_Yes"
EMERGENCY_NONE_COLUMN = "Hospital_Emergency_Room_Visits_None"
EMERGENCY_ONE_COLUMN = "Hospital_Emergency_Room_Visits_1_time"

# Present in the matrix but not features: the SOAK subset label and the
# replication's own outcome. The latter stays out of the extension feature sets
# for now so they match the validated one exactly; adding autism diagnosis as a
# predictor is a variant for the manuscript, not a silent default.
NON_FEATURE_COLUMNS = ("survey_year", "y")


@dataclass(frozen=True)
class OutcomeDefinition:
    """One thing to predict, and what must be removed before predicting it."""

    key: str
    label: str
    positive: pl.Expr
    drop_columns: tuple[str, ...]

    def feature_columns(self, all_columns: list[str]) -> list[str]:
        """Every column that is neither structural nor part of this outcome."""
        excluded = set(NON_FEATURE_COLUMNS) | set(self.drop_columns)
        return [name for name in all_columns if name not in excluded]


FIXTURE_OUTCOMES: dict[str, OutcomeDefinition] = {
    "foregone_care": OutcomeDefinition(
        key="foregone_care",
        label="Foregone care: needed health care and did not receive it",
        positive=pl.col(FOREGONE_CARE_COLUMN) == 1,
        drop_columns=(FOREGONE_CARE_COLUMN,),
    ),
    "ed_any": OutcomeDefinition(
        key="ed_any",
        label="Emergency department use: one or more visits",
        positive=pl.col(EMERGENCY_NONE_COLUMN) == 0,
        drop_columns=(EMERGENCY_NONE_COLUMN, EMERGENCY_ONE_COLUMN),
    ),
    "ed_repeat": OutcomeDefinition(
        key="ed_repeat",
        label="Emergency department use: two or more visits",
        positive=(pl.col(EMERGENCY_NONE_COLUMN) == 0) & (pl.col(EMERGENCY_ONE_COLUMN) == 0),
        drop_columns=(EMERGENCY_NONE_COLUMN, EMERGENCY_ONE_COLUMN),
    ),
    # The replication's own outcome, included so the extension machinery can be
    # pointed at it and reproduce a known answer.
    "autism": OutcomeDefinition(
        key="autism",
        label="Reported autism diagnosis (the replication's outcome)",
        positive=pl.col("y") == "Yes",
        drop_columns=(),
    ),
}


def outcome_or_exit(key: str) -> OutcomeDefinition:
    """Look up an outcome, or raise with the list of names that do work."""
    if key not in FIXTURE_OUTCOMES:
        available = ", ".join(sorted(FIXTURE_OUTCOMES))
        message = f"Unknown outcome {key!r}. Available: {available}"
        raise SystemExit(message)
    return FIXTURE_OUTCOMES[key]
