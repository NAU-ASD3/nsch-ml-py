"""Write the autism prevalence table and its provenance record.

Usage:

    uv run analyses/autism_prevalence.py --raw-dir "$NSCH_RAW"

Reads the nine Census topical files, computes the per-year and pooled
estimates through ``nsch_ml.prevalence``, and writes two files under
``analyses/results/``: the table itself, and a provenance record naming each
input file's SHA-256 so anyone can confirm they are looking at the same
Census release. ``notebooks/autism_prevalence_2016_2024.py`` shows the working
and the checks; this script exists so the numbers are findable in the
repository without running anything.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import polars as pl

from nsch_ml import prevalence

YEARS = tuple(range(2016, 2025))
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def provenance_record(raw_dir: Path) -> pl.DataFrame:
    """One row per input file, with its fingerprint and the package versions."""
    rows = []
    for year in YEARS:
        path = prevalence.find_topical_file(raw_dir, year)
        rows.append(
            {
                "year": year,
                "file": path.name,
                "sha256": prevalence.sha256_of(path),
                "computed_on": datetime.now(tz=UTC).date().isoformat(),
                "nsch_ml": version("nsch-ml"),
                "polars": version("polars"),
                "pyreadstat": version("pyreadstat"),
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="directory holding the unzipped Census topical files for 2016 to 2024",
    )
    parser.add_argument("--min-age", type=int, default=3)
    parser.add_argument("--max-age", type=int, default=17)
    args = parser.parse_args()

    children = prevalence.load_children(args.raw_dir, YEARS, args.min_age, args.max_age)
    table = prevalence.prevalence_table(children, YEARS)

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"autism_prevalence_{YEARS[0]}_{YEARS[-1]}"
    table.write_csv(RESULTS_DIR / f"{stem}.csv")
    provenance_record(args.raw_dir).write_csv(RESULTS_DIR / f"{stem}.provenance.csv")
    print(
        table.select("period", "ever_n", "ever_pct_weighted", "current_n", "current_pct_weighted")
    )


if __name__ == "__main__":
    main()
