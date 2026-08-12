"""Can the validated fold assignment be reused for the new outcomes?

The fold assignment the replication ran on was drawn by mlr3resampling
stratified on (survey year, autism diagnosis). Reusing it for foregone care
and emergency department use is only legitimate if those outcomes happen to be
spread evenly enough across the same folds. Nothing guarantees that: a draw
balanced on one outcome is balanced on another only by luck.

Reuse is worth having. It keeps the new analyses on exactly the partition that
passed validation, so any difference between them and the replication comes
from the outcome rather than from the split. The alternative, drawing fresh
folds with our own splitter, is available and is the long-term path, but it
gives up that comparability.

This script decides the question against a rule fixed before it ran:

  - every (year, fold) cell holds at least MIN_POSITIVES positive cases, and
  - no cell's positive count deviates from its year's per-fold mean by more
    than TOLERANCE.

It also verifies that the fold file lines up with the matrix row for row,
which everything downstream assumes and nothing so far has checked.

Run from the repository root::

    uv run python analyses/check_fixture_fold_balance.py \\
        --fixture "$REPRO/data_Classif/NSCH_autism.csv" \\
        --folds   "$HOME/Documents/NAU/Grad/Research/ADSI/soak_fixture/nsch_autism_folds.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

# The rule, fixed here before any output was seen. A cell thinner than this
# makes a fold's AUC an estimate of almost nothing; a cell further than this
# from its year's average means the draw is carrying outcome structure it was
# never asked to balance.
MIN_POSITIVES_PER_CELL = 5
TOLERANCE = 0.20

FOREGONE_CARE = "Needed_Health_Care_Not_Received_Yes"
EMERGENCY_NONE = "Hospital_Emergency_Room_Visits_None"
EMERGENCY_ONE = "Hospital_Emergency_Room_Visits_1_time"
AUTISM_OUTCOME = "y"

RULE = "=" * 78


def autism_positive_expression(frame: pl.DataFrame) -> pl.Expr:
    """The outcome is stored as Yes/No text or as 0/1; read which, then match."""
    levels = [str(value) for value in frame[AUTISM_OUTCOME].unique().to_list()]
    return pl.col(AUTISM_OUTCOME) == ("Yes" if "Yes" in levels else 1)


def load_matrix_with_folds(fixture_path: Path, folds_path: Path) -> tuple[pl.DataFrame, list[str]]:
    """Attach the fold assignment to the matrix, and check they describe the same rows."""
    matrix = pl.read_csv(
        fixture_path,
        columns=[
            "survey_year",
            AUTISM_OUTCOME,
            FOREGONE_CARE,
            EMERGENCY_NONE,
            EMERGENCY_ONE,
        ],
    ).with_row_index("row_id", offset=1)
    folds = pl.read_csv(folds_path).select(
        pl.col("row_id").cast(pl.UInt32),
        pl.col("test.subset").cast(pl.Utf8).alias("fold_year"),
        pl.col("fold").cast(pl.Int64),
    )

    print(f"  matrix {matrix.height} rows, fold file {folds.height} rows")
    failures: list[str] = []
    if matrix.height != folds.height:
        failures.append(
            f"The matrix has {matrix.height} rows and the fold file {folds.height}. "
            "They cannot describe the same children."
        )
        return matrix, failures

    joined = matrix.join(folds, on="row_id", how="inner")
    if joined.height != matrix.height:
        failures.append(
            f"Joining on row_id kept {joined.height} of {matrix.height} rows, so the fold "
            "file does not cover the matrix one row per child."
        )
        return joined, failures

    # The fold file records which subset each row belongs to. If that disagrees
    # with the matrix's own survey_year, the two files are ordered differently
    # and every downstream count would be silently wrong.
    misaligned = joined.filter(pl.col("fold_year") != pl.col("survey_year").cast(pl.Utf8)).height
    print(f"  rows whose fold-file year disagrees with the matrix: {misaligned}")
    if misaligned:
        failures.append(
            f"{misaligned} rows have a different survey year in the fold file than in the "
            "matrix. The fold assignment does not line up with this matrix."
        )
    return joined, failures


def report_outcome_balance(frame: pl.DataFrame, label: str, positive: pl.Expr) -> list[str]:
    """Print positives per (year, fold) and judge the spread against the rule."""
    per_cell = (
        frame.group_by(["survey_year", "fold"])
        .agg(pl.len().alias("children"), positive.sum().alias("positives"))
        .sort(["survey_year", "fold"])
    )
    print(f"\n  {label}")
    failures: list[str] = []
    for year in sorted(per_cell["survey_year"].unique().to_list()):
        year_cells = per_cell.filter(pl.col("survey_year") == year)
        counts = [int(value) for value in year_cells["positives"].to_list()]
        expected = sum(counts) / len(counts)
        deviations = [
            abs(count - expected) / expected if expected else float("inf") for count in counts
        ]
        print(
            f"    {year}: {sum(counts)} positives over {len(counts)} folds, "
            f"expected {expected:.1f} per fold"
        )
        print(
            f"      per fold {counts}, "
            f"smallest {min(counts)}, largest deviation {max(deviations):.1%}"
        )
        if min(counts) < MIN_POSITIVES_PER_CELL:
            failures.append(
                f"{label}, {year}: a fold holds only {min(counts)} positive cases, "
                f"below the floor of {MIN_POSITIVES_PER_CELL}."
            )
        if max(deviations) > TOLERANCE:
            failures.append(
                f"{label}, {year}: a fold's positive count sits {max(deviations):.1%} from the "
                f"per-fold average, beyond the {TOLERANCE:.0%} tolerance."
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--folds", required=True)
    args = parser.parse_args()

    fixture_path, folds_path = Path(args.fixture), Path(args.folds)
    for label, path in (("fixture", fixture_path), ("folds", folds_path)):
        if not path.is_file():
            print(f"REFUSED: {label} not found at {path}")
            return 1

    print(RULE)
    print("Alignment")
    frame, failures = load_matrix_with_folds(fixture_path, folds_path)
    if failures:
        print(RULE)
        for failure in failures:
            print(f"REFUSED: {failure}")
        return 1

    any_emergency_visit = pl.col(EMERGENCY_NONE) == 0
    repeat_emergency_visits = any_emergency_visit & (pl.col(EMERGENCY_ONE) == 0)

    print(RULE)
    print("Positive cases per (survey year, fold)")
    print(f"floor {MIN_POSITIVES_PER_CELL} per cell, tolerance {TOLERANCE:.0%} from the mean")

    # The autism outcome is the one the draw was stratified on. It is included
    # as a reference: it shows what a balanced outcome looks like here, which
    # gives the other three something to be compared against.
    failures += report_outcome_balance(
        frame,
        "autism diagnosis (what the draw was stratified on)",
        autism_positive_expression(frame),
    )
    failures += report_outcome_balance(frame, "foregone care (k4q27)", pl.col(FOREGONE_CARE) == 1)
    failures += report_outcome_balance(frame, "ED use, one or more visits", any_emergency_visit)
    failures += report_outcome_balance(frame, "ED use, two or more visits", repeat_emergency_visits)

    print("\n" + RULE)
    if failures:
        for failure in failures:
            print(f"REFUSED: {failure}")
        print(
            f"\n{len(failures)} check(s) failed. Do not reuse this fold assignment for the "
            "outcome named above; draw fresh folds stratified on it instead."
        )
        return 1
    print("All outcomes are adequately balanced across the existing folds.")
    print("Reuse is justified. Record this output alongside the analysis plan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
