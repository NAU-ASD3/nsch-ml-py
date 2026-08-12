"""Draw a fold assignment for one outcome, and record where it came from.

The replication ran on folds drawn in R, stratified on survey year and autism
diagnosis. Those folds are balanced on that outcome and, with respect to any
other outcome, are an ordinary unstratified split. That is valid, but
stratifying on the outcome you are actually predicting is better, and it
matters most where the positive cases are thinnest, which is exactly where
these new outcomes sit.

So each extension analysis gets its own assignment, drawn here by the splitter
in ``nsch_ml.soak``, which deals every (subset, outcome) cell across the folds
so per-cell counts differ by at most one.

Nothing about a fold assignment is interesting unless you can say how it was
produced. Alongside the folds this writes a provenance file naming the seed,
the fold count, the outcome, the checksum of the matrix it was drawn against,
and the versions of everything involved. A results file whose folds cannot be
traced is not reproducible, however careful the modelling was.

Run from the repository root::

    uv run python analyses/draw_outcome_folds.py \\
        --matrix fixture \\
        --fixture "$REPRO/data_Classif/NSCH_autism.csv" \\
        --outcome foregone_care \\
        --out analyses/folds/fixture_foregone_care_folds.csv
"""

from __future__ import annotations

import argparse
import hashlib
import platform
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import polars as pl
from outcomes import matrix_or_exit

from nsch_ml.soak import assign_folds

DEFAULT_FOLDS = 10
# Seed 1 throughout the project, matching the replication's convention.
DEFAULT_SEED = 1
TRACKED_PACKAGES = ("nsch-ml", "polars", "numpy", "scikit-learn")

RULE = "=" * 78


def file_md5(path: Path) -> str:
    """Checksum of the matrix, so a fold file can name what it was drawn against."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, help="fixture or service_use")
    parser.add_argument("--fixture", required=True, help="path to the matrix CSV")
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--out", required=True, help="where to write the fold assignment")
    parser.add_argument(
        "--provenance",
        default=None,
        help="where to write the provenance record; defaults to <out>.provenance.csv",
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    if not fixture_path.is_file():
        print(f"REFUSED: no matrix at {fixture_path}")
        return 1
    spec = matrix_or_exit(args.matrix)
    outcome = spec.outcome_or_exit(args.outcome)
    subset_column = spec.subset_column

    matrix = pl.read_csv(fixture_path)
    missing = [
        name
        for name in (subset_column, *spec.non_feature_columns, *outcome.drop_columns)
        if name not in matrix.columns
    ]
    if missing:
        print(f"REFUSED: the matrix is missing {', '.join(missing)}")
        return 1

    positive = matrix.select(outcome.positive.alias("positive"))["positive"]
    subset = matrix[subset_column].cast(pl.Utf8)
    checksum = file_md5(fixture_path)

    print(RULE)
    print(f"Matrix    {fixture_path}")
    print(f"          {spec.label}")
    print(f"md5       {checksum}")
    print(f"Outcome   {outcome.key}: {outcome.label}")
    print(f"Rows      {matrix.height}, positives {int(positive.sum())}")
    print(
        f"Draw      {args.folds} folds, seed {args.seed}, stratified on ({subset_column}, outcome)"
    )

    fold_ids = assign_folds(
        subset=subset,
        outcome=positive,
        n_folds=args.folds,
        seed=args.seed,
    )

    assignment = pl.DataFrame(
        {
            "row_id": np.arange(1, matrix.height + 1, dtype=np.int64),
            "test.subset": subset,
            "fold": fold_ids,
        }
    )

    # The splitter guarantees per-cell counts within one of each other. Checking
    # it here costs nothing and turns a guarantee into an observation.
    print(RULE)
    print("Positives per (survey year, fold)")
    per_cell = (
        assignment.with_columns(positive.alias("positive"))
        .group_by(["test.subset", "fold"])
        .agg(pl.col("positive").sum().alias("positives"))
        .sort(["test.subset", "fold"])
    )
    failures: list[str] = []
    for year in sorted(per_cell["test.subset"].unique().to_list()):
        counts = [
            int(value)
            for value in per_cell.filter(pl.col("test.subset") == year)["positives"].to_list()
        ]
        print(f"  {year}: {counts}  (spread {max(counts) - min(counts)})")
        if max(counts) - min(counts) > 1:
            failures.append(
                f"{year}: fold positive counts range from {min(counts)} to {max(counts)}. "
                "Stratification deals each cell round robin, so a spread above one means "
                "the assignment is not what the splitter promises."
            )
    if failures:
        print(RULE)
        for failure in failures:
            print(f"REFUSED: {failure}")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assignment.write_csv(out_path)

    provenance_path = (
        Path(args.provenance) if args.provenance else out_path.with_suffix(".provenance.csv")
    )
    provenance = pl.DataFrame(
        {
            "field": [
                "drawn_at_utc",
                "matrix_key",
                "outcome",
                "outcome_label",
                "matrix",
                "matrix_md5",
                "rows",
                "positives",
                "n_folds",
                "seed",
                "stratified_on",
                "dropped_from_features",
                "python",
                *TRACKED_PACKAGES,
            ],
            "value": [
                datetime.now(UTC).isoformat(timespec="seconds"),
                spec.key,
                outcome.key,
                outcome.label,
                str(fixture_path),
                checksum,
                str(matrix.height),
                str(int(positive.sum())),
                str(args.folds),
                str(args.seed),
                f"({subset_column}, outcome)",
                ", ".join(outcome.drop_columns) or "none",
                platform.python_version(),
                *[package_version(name) for name in TRACKED_PACKAGES],
            ],
        }
    )
    provenance.write_csv(provenance_path)

    print(RULE)
    print(f"wrote {out_path}")
    print(f"wrote {provenance_path}")
    print("Both belong in the same commit as any results drawn from them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
