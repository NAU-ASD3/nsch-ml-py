"""Characterize the 2016-2023 service-use matrix before any model reads it.

Prints row counts, identifier structure, period coding, outcome prevalence, and
design-weight completeness for ``2016_2023_ServiceUse.csv``, and exits nonzero
when anything it finds contradicts an assumption the extension analysis rests
on. Nothing here fits a model or writes a file; the point is that every number
about this matrix appears in a terminal before it appears in a document.

The matrix is not in this repository. Point ``MONSOON_OLD`` at the directory
holding it:

    export MONSOON_OLD="$HOME/Documents/NAU/Grad/Research/ADSI/Monsoon - ASD3 ML Old"
    uv run python analyses/characterize_service_use.py
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import polars as pl

MATRIX_FILENAME = "2016_2023_ServiceUse.csv"

# Identifier and survey-design columns. These are excluded from any feature set
# and must all be present for the fold draw and the later weighted work.
ID_COLUMNS = ("hhid", "stratum", "fwc", "fipsst", "state", "period")

# Positive-class definitions. Foregone care and behaviour therapy are direct
# "Yes" indicators. ED use is the complement of the "no visits" indicator: a
# positive case is a child with one or more emergency department visits, so the
# "1 time" column folds into the positive class rather than standing alone.
FOREGONE_CARE = "k4q27=Yes"
ED_NONE = "hospitaler=None"
ED_ONE = "hospitaler=1 time"
BEHAVIOR_THERAPY = "autismtreat=Yes"

OUTCOME_COLUMNS = (FOREGONE_CARE, ED_NONE, ED_ONE, BEHAVIOR_THERAPY)

# Questions the survey asks only about children who already have an autism
# diagnosis. Their coverage is the evidence for or against the claim that this
# matrix holds the autism subset rather than the full child population.
AUTISM_ONLY_COLUMNS = (
    "k2q35a_1_years",
    "k2q35b=Yes",
    "k2q35c=Mild",
    "k2q35c=Moderate",
    "k2q35d=Primary Care Provider",
)

# If this matrix were the full child population, the autism screener itself
# would vary and would appear as a one-hot column. Its absence is evidence for
# the autism-subset reading, so the script says so either way.
AUTISM_SCREENER = "k2q35a=Yes"

# Substrings whose columns are printed for the leak audit. Each target's own
# columns are deleted before fitting; these are the neighbours that need a
# written verdict because they may be definitionally entangled with a target
# rather than merely correlated with it.
LEAK_SCAN_TOKENS = (
    "k4q27",
    "hospitaler",
    "autismtreat",
    "k4q20r",
    "k4q02_r",
    "k4q22_r",
    "k4q24_r",
)

# Fold-count rule from the analysis plan: ten folds by default, five when ten
# would leave a (period, fold) cell with fewer than five positive cases. A task
# too thin even for five folds is a finding, not a configuration detail.
MIN_POSITIVES_PER_CELL = 5
DEFAULT_FOLDS = 10
FALLBACK_FOLDS = 5

RULE = "=" * 78


def resolve_matrix_path() -> Path | None:
    """Locate the matrix from ``MONSOON_OLD``, or explain why it cannot."""
    raw = os.environ.get("MONSOON_OLD")
    if not raw:
        print("MONSOON_OLD is not set. Point it at the directory holding the matrix:")
        print('  export MONSOON_OLD="$HOME/Documents/NAU/Grad/Research/ADSI/Monsoon - ASD3 ML Old"')
        return None
    path = Path(raw).expanduser() / MATRIX_FILENAME
    if not path.is_file():
        print(f"No file at {path}")
        return None
    return path


def file_md5(path: Path) -> str:
    """Return the md5 of a file, read in chunks so size does not matter."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_missing_columns(frame: pl.DataFrame) -> list[str]:
    """Refuse on any absent identifier or outcome column."""
    present = set(frame.columns)
    missing = [name for name in (*ID_COLUMNS, *OUTCOME_COLUMNS) if name not in present]
    if missing:
        return [f"Required column absent: {name}" for name in missing]
    return []


def report_outcome_coding(frame: pl.DataFrame) -> list[str]:
    """Print dtype, nulls and distinct values per outcome column; refuse on surprises."""
    failures: list[str] = []
    for name in OUTCOME_COLUMNS:
        dtype = frame.schema[name]
        nulls = int(frame[name].null_count())
        distinct = sorted(str(value) for value in frame[name].unique().to_list())
        shown = ", ".join(distinct[:6])
        print(f"  {name}")
        print(f"    dtype {dtype}, nulls {nulls}, distinct values [{shown}]")
        if not dtype.is_numeric():
            failures.append(
                f"{name} has dtype {dtype}, not numeric, so its 0/1 coding cannot be trusted."
            )
        if nulls:
            failures.append(
                f"{name} has {nulls} null values. The positive-class rule is undefined for "
                "them, so the analysis plan needs an explicit missing-data decision before "
                "this outcome is modelled."
            )
    return failures


def report_identifiers(frame: pl.DataFrame) -> None:
    """Print household identifier structure, which decides whether grouping matters."""
    rows = frame.height
    unique = frame["hhid"].n_unique()
    repeated = frame.group_by("hhid").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    cross_period = (
        frame.group_by("hhid")
        .agg(pl.col("period").n_unique().alias("periods"))
        .filter(pl.col("periods") > 1)
    )
    print(f"  rows {rows}, distinct hhid {unique}")
    print(f"  hhid values appearing more than once: {repeated.height}")
    print(f"  hhid values spanning more than one period: {cross_period.height}")
    if unique == rows:
        print("  hhid is unique per row, so household grouping in the fold draw")
        print("  reduces to row grouping. Record it as a deliberate no-op.")
    else:
        print("  hhid repeats. NSCH is cross-sectional, so these are identifier")
        print("  collisions to understand before the fold draw, not panel households.")


def report_periods(frame: pl.DataFrame) -> list[str]:
    """Print the period coding and per-period row counts; refuse on an empty period."""
    counts = frame.group_by("period").agg(pl.len().alias("rows")).sort("period")
    for row in counts.iter_rows(named=True):
        print(f"  period {row['period']}: {row['rows']} rows")
    empty = counts.filter(pl.col("rows") == 0)
    if empty.height:
        return [f"{empty.height} period(s) have no rows."]
    return []


def report_weights(frame: pl.DataFrame) -> list[str]:
    """Print completeness and range of the child weight."""
    nulls = int(frame["fwc"].null_count())
    print(f"  fwc nulls {nulls} of {frame.height}")
    if not frame.schema["fwc"].is_numeric():
        return [f"fwc has dtype {frame.schema['fwc']}, not numeric."]
    stats = frame.select(
        pl.col("fwc").min().alias("min"),
        pl.col("fwc").max().alias("max"),
        pl.col("fwc").sum().alias("sum"),
    ).row(0, named=True)
    print(f"  fwc min {stats['min']}, max {stats['max']}, sum {stats['sum']}")
    if nulls:
        return ["fwc is incomplete, which blocks the weighted analysis on this matrix."]
    return []


def report_autism_subset_evidence(frame: pl.DataFrame) -> list[str]:
    """Test the claim that this matrix holds only children with an autism diagnosis."""
    present = set(frame.columns)
    if AUTISM_SCREENER in present:
        print(f"  {AUTISM_SCREENER} is PRESENT, which argues against the autism-subset reading")
    else:
        print(f"  {AUTISM_SCREENER} is absent, consistent with a constant column being dropped")

    coverage: float | None = None
    for name in AUTISM_ONLY_COLUMNS:
        if name not in present:
            print(f"  {name}: absent")
            continue
        non_null = frame.height - int(frame[name].null_count())
        share = non_null / frame.height if frame.height else 0.0
        print(f"  {name}: {non_null} non-null of {frame.height} ({share:.1%})")
        if name == "k2q35a_1_years":
            coverage = share

    if coverage is None:
        return ["k2q35a_1_years is absent, so the autism-subset claim cannot be tested here."]
    if coverage < 0.50:
        return [
            f"k2q35a_1_years is populated for only {coverage:.1%} of rows. That is the "
            "signature of a full-population sample, which contradicts the extension design's "
            "assumption that this matrix is the autism subset."
        ]
    return []


def report_leak_candidates(frame: pl.DataFrame) -> None:
    """Print every column whose name touches an outcome, for the written leak audit."""
    for token in LEAK_SCAN_TOKENS:
        matches = [name for name in frame.columns if token in name]
        joined = "; ".join(matches) if matches else "none"
        print(f"  {token}: {joined}")


def report_three_state(frame: pl.DataFrame, none_col: str, one_col: str, label: str) -> list[str]:
    """Print the level structure implied by a pair of one-hot columns.

    The encoder that built this matrix dropped one level per categorical, so a
    three-level variable leaves two columns and the third level is the rows
    where both are zero. Recovering it matters because that residual is what a
    two-or-more-visits outcome would be built from, and because hospitaler
    gained a fourth survey level in 2022. If the harmonization folded the new
    top levels together, the residual share should move smoothly across the
    periods; a jump at the last period would say it did not.
    """
    absent = [name for name in (none_col, one_col) if name not in frame.columns]
    if absent:
        print(f"  {label}: absent ({', '.join(absent)})")
        return []
    both = frame.filter((pl.col(none_col) == 1) & (pl.col(one_col) == 1)).height
    if both:
        return [
            f"{label}: {both} rows set both {none_col} and {one_col}, so the pair is not a "
            "valid one-hot encoding and no level can be recovered from it."
        ]
    summary = (
        frame.group_by("period")
        .agg(
            pl.len().alias("rows"),
            (pl.col(none_col) == 1).sum().alias("lowest"),
            (pl.col(one_col) == 1).sum().alias("middle"),
            ((pl.col(none_col) == 0) & (pl.col(one_col) == 0)).sum().alias("residual"),
        )
        .sort("period")
    )
    print(f"  {label}: {none_col} / {one_col} / residual")
    for row in summary.iter_rows(named=True):
        total = row["rows"]
        print(
            f"    period {row['period']}: "
            f"{row['lowest']} ({row['lowest'] / total:.1%}), "
            f"{row['middle']} ({row['middle'] / total:.1%}), "
            f"{row['residual']} ({row['residual'] / total:.1%})"
        )
    return []


def report_prevalence(frame: pl.DataFrame) -> list[str]:
    """Print per-period positives and prevalence per outcome; refuse on a dead cell."""
    targets: tuple[tuple[str, pl.Expr], ...] = (
        ("foregone care (k4q27)", pl.col(FOREGONE_CARE) == 1),
        (f"ED use (one or more visits, {ED_NONE} == 0)", pl.col(ED_NONE) == 0),
        (
            "repeat ED use (two or more visits, both hospitaler one-hots zero)",
            (pl.col(ED_NONE) == 0) & (pl.col(ED_ONE) == 0),
        ),
        ("behaviour therapy (autismtreat)", pl.col(BEHAVIOR_THERAPY) == 1),
    )
    failures: list[str] = []
    for label, positive in targets:
        summary = (
            frame.group_by("period")
            .agg(pl.len().alias("rows"), positive.sum().alias("positives"))
            .with_columns((pl.col("positives") / pl.col("rows")).alias("prevalence"))
            .sort("period")
        )
        print(f"  {label}")
        for row in summary.iter_rows(named=True):
            print(
                f"    period {row['period']}: {row['positives']} of {row['rows']} "
                f"({row['prevalence']:.1%})"
            )
        # Series.min() is typed as a wide union covering every dtype polars can
        # hold, so it does not narrow to int. Going through to_list() keeps the
        # arithmetic honest without a cast that would suppress a real error.
        per_period = [int(value) for value in summary["positives"].to_list()]
        least = min(per_period) if per_period else 0
        if least == 0:
            failures.append(f"{label} has a period with zero positive cases.")
            continue
        if least < MIN_POSITIVES_PER_CELL * FALLBACK_FOLDS:
            failures.append(
                f"{label} has a period with only {least} positive cases, too few for even "
                f"{FALLBACK_FOLDS} folds at {MIN_POSITIVES_PER_CELL} positives per cell."
            )
            continue
        folds = DEFAULT_FOLDS if least >= MIN_POSITIVES_PER_CELL * DEFAULT_FOLDS else FALLBACK_FOLDS
        print(f"    thinnest period has {least} positives, so use {folds} folds")
    return failures


def main() -> int:
    path = resolve_matrix_path()
    if path is None:
        return 1

    print(RULE)
    print(f"Matrix   {path}")
    print(f"md5      {file_md5(path)}")

    frame = pl.read_csv(path, infer_schema_length=None)
    print(f"Shape    {frame.height} rows, {frame.width} columns")

    failures = report_missing_columns(frame)
    if failures:
        print(RULE)
        for failure in failures:
            print(f"REFUSED: {failure}")
        return 1

    print(RULE)
    print("Household identifiers")
    report_identifiers(frame)

    print(RULE)
    print("Periods")
    failures += report_periods(frame)

    print(RULE)
    print("Outcome coding")
    failures += report_outcome_coding(frame)

    print(RULE)
    print("Autism-subset evidence")
    failures += report_autism_subset_evidence(frame)

    print(RULE)
    print("Survey weight")
    failures += report_weights(frame)

    print(RULE)
    print("Leak-audit candidates (columns touching an outcome)")
    report_leak_candidates(frame)

    print(RULE)
    print("Implied level structure behind the one-hot pairs")
    failures += report_three_state(frame, ED_NONE, ED_ONE, "hospitaler")
    failures += report_three_state(frame, "k4q20r=0 visits", "k4q20r=1 visit", "k4q20r")

    if not any(f.startswith(("k4q27", "hospitaler", "autismtreat")) for f in failures):
        print(RULE)
        print("Outcome prevalence by period")
        failures += report_prevalence(frame)

    print(RULE)
    if failures:
        for failure in failures:
            print(f"REFUSED: {failure}")
        print(f"{len(failures)} check(s) failed. No number here goes into a document.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
