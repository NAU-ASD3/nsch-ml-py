"""The outcome and matrix definitions the extension analyses share.

Three things have to agree about an outcome: the fold draw stratifies on it,
the model fits it, and the leak audit decides which columns to remove because
they would give it away. If those three read the definition from three places,
they will eventually disagree, and the failure is silent: the run completes and
the numbers look plausible.

So each outcome is defined once, here, and every script imports it.

Two matrices carry these outcomes and they are not interchangeable. The
full-population matrix spans two survey years, names its columns in prose, and
splits on ``survey_year``. The autism-subset matrix spans four two-year
periods, names its columns by survey code, and splits on ``period``. An
outcome therefore belongs to a matrix, and asking for one on the wrong matrix
is an error rather than something to paper over.

The positive-class rules follow docs/extension-analysis-plan.md:

  foregone care     the child needed health care in the past year and did not
                    receive it
  ED use, any       one or more emergency room visits, which is the complement
                    of the "no visits" indicator
  ED use, repeat    two or more visits. The survey records visits in three
                    bands and each matrix carries indicators for the lowest
                    two, so "two or more" is the rows where both are zero
  behaviour therapy the child received behavioural treatment for autism, asked
                    only of children who already have a diagnosis

Only an outcome's own columns are dropped from the features. Neighbouring
columns that merely correlate with an outcome stay in; predicting an outcome
from its correlates is the analysis, not a leak. The one column that looked
like a leak on inspection, ``k4q20r``, turned out to count preventive visits
rather than all visits, so it cannot imply anything about emergency use and it
stays.

The ``_strict`` variants are the pre-registered sensitivity analysis for
foregone care. ``k4q22_r`` and ``k4q24_r`` carry "No, but this child needed to
see..." levels, which are specific instances of the same construct the outcome
measures globally. They stay in the primary analysis because removing every
correlate leaves nothing to predict from, and the variant exists so the
question is answered with a number rather than an argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, eq=False)
class OutcomeDefinition:
    """One thing to predict, and what must be removed before predicting it."""

    key: str
    label: str
    positive: pl.Expr
    drop_columns: tuple[str, ...]
    #: Which outcome's fold assignment this uses. A sensitivity variant shares
    #: its base outcome's positive-class rule, so it shares the folds drawn for
    #: it; naming that here lets a runner verify it is using the right file
    #: rather than inferring it from a name.
    folds_from: str | None = None

    @property
    def folds_key(self) -> str:
        return self.folds_from or self.key


@dataclass(frozen=True, eq=False)
class MatrixSpec:
    """One design matrix: how to split it, what is not a feature, what it can predict."""

    key: str
    label: str
    filename: str
    subset_column: str
    #: Identifier, design and outcome-of-another-analysis columns. Never features.
    non_feature_columns: tuple[str, ...]
    outcomes: dict[str, OutcomeDefinition]

    def feature_columns(self, all_columns: list[str], outcome: OutcomeDefinition) -> list[str]:
        """Every column that is neither structural nor part of this outcome."""
        excluded = set(self.non_feature_columns) | set(outcome.drop_columns)
        return [name for name in all_columns if name not in excluded]

    def outcome_or_exit(self, key: str) -> OutcomeDefinition:
        if key not in self.outcomes:
            available = ", ".join(sorted(self.outcomes))
            message = f"Matrix {self.key!r} has no outcome {key!r}. Available: {available}"
            raise SystemExit(message)
        return self.outcomes[key]


# --------------------------------------------------------------------------
# Full-population matrix: 46,010 children, survey years 2019 and 2020.
# --------------------------------------------------------------------------

FIXTURE_FOREGONE = "Needed_Health_Care_Not_Received_Yes"
FIXTURE_ED_NONE = "Hospital_Emergency_Room_Visits_None"
FIXTURE_ED_ONE = "Hospital_Emergency_Room_Visits_1_time"

# The full-population counterparts of k4q22_r and k4q24_r. Both levels of each,
# not only the "No, but needed" one: dropping a single level of a one-hot leaves
# the information recoverable from the others.
FIXTURE_UNMET_NEED_NEIGHBOURS = (
    "Mental_Health_Professional_Treatment_Yes",
    "Mental_Health_Professional_Treatment_No__but_this_child_needed_to_see_"
    "a_mental_health_professional",
    "Specialist_Visit_Yes",
    "Specialist_Visit_No__but_this_child_needed_to_see_a_specialist",
)

# Questions about the process of obtaining care in the same twelve-month window
# as an access outcome, identified by the feature audit and listed in the
# 24 August amendment to docs/extension-analysis-plan.md. Excluded in the
# `_conservative` variants only; the primary specifications keep them.
FIXTURE_CARE_SEEKING = (
    "Frustrated_In_Efforts_to_Get_Service_Never",
    "Frustrated_In_Efforts_to_Get_Service_Sometimes",
    "Frustrated_In_Efforts_to_Get_Service_Usually",
    "Need_a_Referral_Yes",
    "Health_Insurance___Benefits_Cover_Services_Always",
    "Health_Insurance___Benefits_Cover_Services_Usually",
    "Health_Insurance___Benefits_Cover_Services_Sometimes",
    "Health_Insurance___Allow_to_See_Provider_Always",
    "Health_Insurance___Allow_to_See_Provider_Usually",
    "Health_Insurance___Allow_to_See_Provider_Sometimes",
    "Health_Insurance___Cover_Mental_Behavioral_Needs_Always",
    "Health_Insurance___Cover_Mental_Behavioral_Needs_Usually",
    "Health_Insurance___Cover_Mental_Behavioral_Needs_Sometimes",
)

FIXTURE = MatrixSpec(
    key="fixture",
    label="Full child population, survey years 2019 and 2020",
    filename="NSCH_autism.csv",
    subset_column="survey_year",
    # y is the replication's outcome. Keeping it out leaves the feature set
    # identical to the validated one; adding autism diagnosis as a predictor is
    # a variant for the manuscript, not a silent default.
    non_feature_columns=("survey_year", "y"),
    outcomes={
        "foregone_care": OutcomeDefinition(
            key="foregone_care",
            label="Foregone care: needed health care and did not receive it",
            positive=pl.col(FIXTURE_FOREGONE) == 1,
            drop_columns=(FIXTURE_FOREGONE,),
        ),
        "foregone_care_strict": OutcomeDefinition(
            key="foregone_care_strict",
            label="Foregone care, with the specific unmet-need items removed",
            positive=pl.col(FIXTURE_FOREGONE) == 1,
            drop_columns=(FIXTURE_FOREGONE, *FIXTURE_UNMET_NEED_NEIGHBOURS),
            folds_from="foregone_care",
        ),
        "ed_any": OutcomeDefinition(
            key="ed_any",
            label="Emergency department use: one or more visits",
            positive=pl.col(FIXTURE_ED_NONE) == 0,
            drop_columns=(FIXTURE_ED_NONE, FIXTURE_ED_ONE),
        ),
        "ed_repeat": OutcomeDefinition(
            key="ed_repeat",
            label="Emergency department use: two or more visits",
            positive=(pl.col(FIXTURE_ED_NONE) == 0) & (pl.col(FIXTURE_ED_ONE) == 0),
            drop_columns=(FIXTURE_ED_NONE, FIXTURE_ED_ONE),
        ),
        # The replication's own outcome. Not part of the extension; it is here
        # so the extension runner can be pointed at a question whose answer is
        # already known and checked against it.
        "autism": OutcomeDefinition(
            key="autism",
            label="Reported autism diagnosis (the replication's outcome)",
            positive=pl.col("y") == "Yes",
            drop_columns=(),
        ),
        "foregone_care_conservative": OutcomeDefinition(
            key="foregone_care_conservative",
            label="Foregone care, with care-seeking-process features removed",
            positive=pl.col(FIXTURE_FOREGONE) == 1,
            drop_columns=(FIXTURE_FOREGONE, *FIXTURE_CARE_SEEKING),
            folds_from="foregone_care",
        ),
        "ed_any_conservative": OutcomeDefinition(
            key="ed_any_conservative",
            label="ED use, one or more visits, care-seeking-process features removed",
            positive=pl.col(FIXTURE_ED_NONE) == 0,
            drop_columns=(FIXTURE_ED_NONE, FIXTURE_ED_ONE, *FIXTURE_CARE_SEEKING),
            folds_from="ed_any",
        ),
        "ed_repeat_conservative": OutcomeDefinition(
            key="ed_repeat_conservative",
            label="ED use, two or more visits, care-seeking-process features removed",
            positive=(pl.col(FIXTURE_ED_NONE) == 0) & (pl.col(FIXTURE_ED_ONE) == 0),
            drop_columns=(FIXTURE_ED_NONE, FIXTURE_ED_ONE, *FIXTURE_CARE_SEEKING),
            folds_from="ed_repeat",
        ),
    },
)


# --------------------------------------------------------------------------
# Autism-subset matrix: 6,088 children with autism, four two-year periods.
# --------------------------------------------------------------------------

SERVICE_FOREGONE = "k4q27=Yes"
SERVICE_ED_NONE = "hospitaler=None"
SERVICE_ED_ONE = "hospitaler=1 time"
SERVICE_THERAPY = "autismtreat=Yes"

# Both levels of each variable, not only the "No, but needed" one. Dropping a
# single level of a one-hot leaves the information recoverable from the others.
SERVICE_UNMET_NEED_NEIGHBOURS = (
    "k4q22_r=Yes",
    "k4q22_r=No, but this child needed to see a mental health professional",
    "k4q24_r=Yes",
    "k4q24_r=No, but this child needed to see a specialist",
)

SERVICE_CARE_SEEKING = (
    "c4q04=Never",
    "c4q04=Sometimes",
    "c4q04=Usually",
    "k5q10=Yes",
    "k5q11=Not difficult",
    "k5q11=Somewhat difficult",
    "k5q11=Very difficult",
    "k5q20_r=No",
    "k5q20_r=Yes",
    "k5q21=Yes",
)

SERVICE_USE = MatrixSpec(
    key="service_use",
    label="Children with autism, survey years 2016 to 2023",
    filename="2016_2023_ServiceUse.csv",
    subset_column="period",
    # hhid is unique per row and carries no information; stratum, fwc, fipsst
    # and state are survey design columns, and period is the subset label.
    non_feature_columns=("hhid", "stratum", "fwc", "fipsst", "state", "period"),
    outcomes={
        "foregone_care": OutcomeDefinition(
            key="foregone_care",
            label="Foregone care: needed health care and did not receive it",
            positive=pl.col(SERVICE_FOREGONE) == 1,
            drop_columns=(SERVICE_FOREGONE,),
        ),
        "foregone_care_strict": OutcomeDefinition(
            key="foregone_care_strict",
            label="Foregone care, with the specific unmet-need items removed",
            positive=pl.col(SERVICE_FOREGONE) == 1,
            drop_columns=(SERVICE_FOREGONE, *SERVICE_UNMET_NEED_NEIGHBOURS),
            folds_from="foregone_care",
        ),
        "ed_any": OutcomeDefinition(
            key="ed_any",
            label="Emergency department use: one or more visits",
            positive=pl.col(SERVICE_ED_NONE) == 0,
            drop_columns=(SERVICE_ED_NONE, SERVICE_ED_ONE),
        ),
        "ed_repeat": OutcomeDefinition(
            key="ed_repeat",
            label="Emergency department use: two or more visits",
            positive=(pl.col(SERVICE_ED_NONE) == 0) & (pl.col(SERVICE_ED_ONE) == 0),
            drop_columns=(SERVICE_ED_NONE, SERVICE_ED_ONE),
        ),
        "behaviour_therapy": OutcomeDefinition(
            key="behaviour_therapy",
            label="Behaviour therapy received for autism",
            positive=pl.col(SERVICE_THERAPY) == 1,
            drop_columns=(SERVICE_THERAPY,),
        ),
        "foregone_care_conservative": OutcomeDefinition(
            key="foregone_care_conservative",
            label="Foregone care, with care-seeking-process features removed",
            positive=pl.col(SERVICE_FOREGONE) == 1,
            drop_columns=(SERVICE_FOREGONE, *SERVICE_CARE_SEEKING),
            folds_from="foregone_care",
        ),
        "ed_any_conservative": OutcomeDefinition(
            key="ed_any_conservative",
            label="ED use, one or more visits, care-seeking-process features removed",
            positive=pl.col(SERVICE_ED_NONE) == 0,
            drop_columns=(SERVICE_ED_NONE, SERVICE_ED_ONE, *SERVICE_CARE_SEEKING),
            folds_from="ed_any",
        ),
        "ed_repeat_conservative": OutcomeDefinition(
            key="ed_repeat_conservative",
            label="ED use, two or more visits, care-seeking-process features removed",
            positive=(pl.col(SERVICE_ED_NONE) == 0) & (pl.col(SERVICE_ED_ONE) == 0),
            drop_columns=(SERVICE_ED_NONE, SERVICE_ED_ONE, *SERVICE_CARE_SEEKING),
            folds_from="ed_repeat",
        ),
        "behaviour_therapy_conservative": OutcomeDefinition(
            key="behaviour_therapy_conservative",
            label="Behaviour therapy, with care-seeking-process features removed",
            positive=pl.col(SERVICE_THERAPY) == 1,
            drop_columns=(SERVICE_THERAPY, *SERVICE_CARE_SEEKING),
            folds_from="behaviour_therapy",
        ),
    },
)


MATRICES: dict[str, MatrixSpec] = {FIXTURE.key: FIXTURE, SERVICE_USE.key: SERVICE_USE}


def matrix_or_exit(key: str) -> MatrixSpec:
    """Look up a matrix, or raise with the names that do work."""
    if key not in MATRICES:
        available = ", ".join(sorted(MATRICES))
        message = f"Unknown matrix {key!r}. Available: {available}"
        raise SystemExit(message)
    return MATRICES[key]
