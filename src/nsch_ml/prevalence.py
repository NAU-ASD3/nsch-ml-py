"""Autism prevalence straight from the Census NSCH public-use files.

This module exists so that a headline descriptive figure, such as how many
children aged 3 to 17 have ever been diagnosed with autism, can be traced from
the Census Bureau's own file to the reported number with nothing in between.
It reads the raw Stata files and does not pass through the harmonization
pipeline in nsch-py, which also makes it usable as an independent check on
that pipeline.

Two survey items carry the whole calculation. ``K2Q35A`` asks whether a doctor
or other health care provider ever told the respondent that the child has
autism or autism spectrum disorder. ``K2Q35B`` is asked only when the answer
is Yes, and asks whether the child currently has the condition. Both are
parent report.

The survey design follows the Census Bureau's guidance for the NSCH:

* weight ``fwc`` for a single year, and ``fwc`` divided by the number of years
  pooled for a multi-year estimate, which gives an average annual figure;
* strata formed by crossing ``fipsst`` with ``stratum``, after recoding the
  post-2016 value ``"2A"`` to ``"2"``;
* ``hhid`` as the primary sampling unit;
* Taylor series linearization for standard errors, with a logit-transformed
  confidence interval for proportions, which is what the Census SAS example
  requests with ``cl (type=logit)``.

Sources: U.S. Census Bureau, *NSCH Guide to Multi-Year Estimates*
(https://www2.census.gov/programs-surveys/nsch/technical-documentation/methodology/NSCH-Guide-to-Multi-Year-Estimates.pdf)
and *NSCH Analytic Guide*
(https://www2.census.gov/programs-surveys/nsch/technical-documentation/methodology/NSCH-Analytic-Guide.pdf).
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import TYPE_CHECKING

import polars as pl
import pyreadstat

if TYPE_CHECKING:
    from collections.abc import Iterable

#: The only columns this module reads from a topical file. Keeping the list
#: short is deliberate: a reader can hold the whole input in their head.
TOPICAL_COLUMNS: tuple[str, ...] = (
    "year",
    "hhid",
    "fipsst",
    "stratum",
    "fwc",
    "sc_age_years",
    "k2q35a",
    "k2q35b",
)

#: Answer codes for the two autism items, as every year's ``.do`` file defines
#: them. Stata's tagged missing values reach us from pyreadstat as a letter.
CODE_YES = "1"
CODE_NO = "2"
CODE_NO_VALID_RESPONSE = "m"
CODE_LOGICAL_SKIP = "l"

#: The five groups every child falls into, exactly one each. The names double
#: as column headers in the accounting table, so they are written to be read.
GROUP_CURRENT = "ever_yes_current_yes"
GROUP_NOT_CURRENT = "ever_yes_current_no"
GROUP_CURRENT_UNKNOWN = "ever_yes_current_missing"
GROUP_NEVER = "ever_no"
GROUP_EVER_UNKNOWN = "ever_missing"
GROUPS: tuple[str, ...] = (
    GROUP_CURRENT,
    GROUP_NOT_CURRENT,
    GROUP_CURRENT_UNKNOWN,
    GROUP_NEVER,
    GROUP_EVER_UNKNOWN,
)

_Z_95 = NormalDist().inv_cdf(0.975)


@dataclass(frozen=True)
class Estimate:
    """One design-based estimate and its 95% confidence interval."""

    value: float
    se: float
    low: float
    high: float


def sha256_of(path: Path | str) -> str:
    """Return the SHA-256 hex digest of a file, read in one-megabyte chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_topical_file(raw_dir: Path | str, year: int, suffix: str = ".dta") -> Path:
    """Locate one year's topical file anywhere under ``raw_dir``.

    Census file names vary by release (``nsch_2023e_topical.dta`` against
    ``nsch_2016_topical.dta``), so the match is on the year, the word
    ``topical`` and the suffix. Finding zero candidates or more than one is
    an error: silently taking the first of two files is how a wrong number
    starts.

    Parameters
    ----------
    raw_dir
        Directory holding the unzipped Census downloads, in any layout.
    year
        Survey year.
    suffix
        ``".dta"`` for the data file, ``".do"`` for its label script.
    """
    matches = sorted(Path(raw_dir).rglob(f"nsch_{year}*_topical{suffix}"))
    if len(matches) != 1:
        found = [str(match) for match in matches]
        raise FileNotFoundError(
            f"Expected exactly one nsch_{year}*_topical{suffix} under {raw_dir}, "
            f"found {len(matches)}: {found}"
        )
    return matches[0]


def _stata_code(value: object) -> str:
    """Turn one cell from pyreadstat into a short string code.

    With ``user_missing=True`` pyreadstat returns numeric answers as floats and
    tagged missing values as their letter (``"m"``, ``"l"`` and so on). Strings
    pass through, numbers become their integer text, and a system missing
    value becomes ``"sysmis"`` so it can never be mistaken for an answer.

    >>> _stata_code(2.0), _stata_code("l"), _stata_code(float("nan"))
    ('2', 'l', 'sysmis')
    """
    if isinstance(value, str):
        return value.strip()
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "sysmis"
    if math.isnan(number):
        return "sysmis"
    return str(int(number))


def read_topical_year(path: Path | str, year: int) -> pl.DataFrame:
    """Read the design and autism columns from one raw topical file.

    Parameters
    ----------
    path
        A Census topical ``.dta`` file.
    year
        The survey year the caller believes the file holds. The file's own
        ``year`` column must agree, or the read is refused.

    Returns
    -------
    pl.DataFrame
        One row per sampled child. ``k2q35a``, ``k2q35b`` and ``stratum`` are
        string codes, and ``stratum`` has ``"2A"`` recoded to ``"2"``.
    """
    path = Path(path)
    # Column case is not guaranteed across releases, so resolve names from the
    # file's own metadata before asking for them.
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    by_lower: dict[str, str] = {name.lower(): name for name in meta.column_names}
    missing = [column for column in TOPICAL_COLUMNS if column not in by_lower]
    if missing:
        raise KeyError(f"{path.name} lacks required columns: {missing}")

    frame, _ = pyreadstat.read_dta(
        str(path),
        usecols=[by_lower[column] for column in TOPICAL_COLUMNS],
        # Without this flag the four tagged missing types collapse into NaN,
        # and "skipped because the answer above was No" becomes
        # indistinguishable from "did not answer".
        user_missing=True,
    )
    frame = frame.rename(columns={by_lower[column]: column for column in TOPICAL_COLUMNS})

    # The tagged-missing columns arrive as object dtype mixing floats and
    # letters, which Polars cannot ingest directly, so each column is coerced
    # cell by cell. Fifty thousand rows makes this a fraction of a second.
    out = pl.DataFrame(
        {
            "year": [int(value) for value in frame["year"]],
            "hhid": [int(value) for value in frame["hhid"]],
            "fipsst": [int(value) for value in frame["fipsst"]],
            "stratum": [_stata_code(value) for value in frame["stratum"]],
            "fwc": [float(value) for value in frame["fwc"]],
            "sc_age_years": [int(value) for value in frame["sc_age_years"]],
            "k2q35a": [_stata_code(value) for value in frame["k2q35a"]],
            "k2q35b": [_stata_code(value) for value in frame["k2q35b"]],
        }
    ).with_columns(
        # 2016 used strata 1 and 2; from 2017 the second is labelled "2A".
        # Census asks for the recode so a pooled file has two strata per state,
        # not three.
        pl.col("stratum").replace({"2A": "2", "2a": "2"})
    )

    file_years = out["year"].unique().to_list()
    if file_years != [year]:
        raise ValueError(f"{path.name}: expected year {year}, the file says {file_years}")
    if out.height != meta.number_rows:
        raise ValueError(f"{path.name}: read {out.height} rows, header says {meta.number_rows}")
    return out


def classify_autism(children: pl.DataFrame) -> pl.DataFrame:
    """Assign every child to exactly one of the five ``GROUPS``.

    Any answer pattern not anticipated here lands in no group and the function
    refuses. A new code in a future release should stop the analysis, not
    slide into a default bucket.
    """
    ever = pl.col("k2q35a")
    current = pl.col("k2q35b")
    group = (
        pl.when((ever == CODE_YES) & (current == CODE_YES))
        .then(pl.lit(GROUP_CURRENT))
        .when((ever == CODE_YES) & (current == CODE_NO))
        .then(pl.lit(GROUP_NOT_CURRENT))
        .when((ever == CODE_YES) & (current == CODE_NO_VALID_RESPONSE))
        .then(pl.lit(GROUP_CURRENT_UNKNOWN))
        # A No on the first item skips the follow-up by design.
        .when((ever == CODE_NO) & (current == CODE_LOGICAL_SKIP))
        .then(pl.lit(GROUP_NEVER))
        .when(ever == CODE_NO_VALID_RESPONSE)
        .then(pl.lit(GROUP_EVER_UNKNOWN))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    out = children.with_columns(autism_group=group)
    unclassified = out.filter(pl.col("autism_group").is_null())
    if unclassified.height > 0:
        combos = unclassified.group_by(["year", "k2q35a", "k2q35b"]).len().sort("year")
        raise ValueError(f"Unanticipated K2Q35A/K2Q35B combinations:\n{combos}")
    return out


def add_indicators(children: pl.DataFrame, min_age: int, max_age: int) -> pl.DataFrame:
    """Add the 0/1 columns the estimators consume.

    ``in_age`` marks the age band. Each outcome gets a ``*_valid`` column (the
    child belongs in the denominator) and a ``*_yes`` column (the child belongs
    in the numerator). Children outside the band, or without a usable answer,
    stay in the frame with zeros: the variance calculation needs the whole
    sample present to treat the age band as a domain.
    """
    in_age = pl.col("sc_age_years").is_between(min_age, max_age)
    group = pl.col("autism_group")
    ever_answered = [
        GROUP_CURRENT,
        GROUP_NOT_CURRENT,
        GROUP_CURRENT_UNKNOWN,
        GROUP_NEVER,
    ]
    ever_yes = [GROUP_CURRENT, GROUP_NOT_CURRENT, GROUP_CURRENT_UNKNOWN]
    # "Currently" needs both items answered. A child whose parent said Yes to
    # "ever" and left "currently" blank cannot be placed on either side.
    current_answered = [GROUP_CURRENT, GROUP_NOT_CURRENT, GROUP_NEVER]
    return children.with_columns(
        in_age=in_age.cast(pl.Int8),
        ever_valid=(in_age & group.is_in(ever_answered)).cast(pl.Int8),
        ever_yes=(in_age & group.is_in(ever_yes)).cast(pl.Int8),
        current_valid=(in_age & group.is_in(current_answered)).cast(pl.Int8),
        current_yes=(in_age & (group == GROUP_CURRENT)).cast(pl.Int8),
        stratum_id=pl.format("{}-{}", pl.col("fipsst"), pl.col("stratum")),
    )


def load_children(
    raw_dir: Path | str, years: Iterable[int], min_age: int = 3, max_age: int = 17
) -> pl.DataFrame:
    """Read, classify and flag every sampled child for the given years."""
    frames = [read_topical_year(find_topical_file(raw_dir, year), year) for year in years]
    return add_indicators(classify_autism(pl.concat(frames)), min_age, max_age)


def _linearized_se(design: pl.DataFrame, score: pl.Expr) -> float:
    """Standard error of a total of ``score`` under stratified PSU sampling.

    This is the with-replacement Taylor linearization that Stata's ``svy``
    and SAS ``PROC SURVEYFREQ`` use by default:
    ``V = sum over strata of n_h / (n_h - 1) * sum over PSUs of (z_hi - zbar_h)^2``,
    where ``z_hi`` is a PSU total. A stratum with one PSU contributes zero,
    which matches the Census example code's ``singleunit(certainty)``.
    """
    psu_totals = (
        design.with_columns(_score=score)
        .group_by(["stratum_id", "hhid"])
        .agg(pl.col("_score").sum())
    )
    per_stratum = psu_totals.group_by("stratum_id").agg(
        n=pl.len(),
        ss=((pl.col("_score") - pl.col("_score").mean()) ** 2).sum(),
    )
    variance = per_stratum.select(
        pl.when(pl.col("n") > 1)
        .then(pl.col("n") / (pl.col("n") - 1) * pl.col("ss"))
        .otherwise(0.0)
        .sum()
    ).item()
    return math.sqrt(float(variance))


def weighted_proportion(design: pl.DataFrame, yes: str, valid: str, weight: str) -> Estimate:
    """Weighted proportion of ``yes`` among ``valid``, with a logit 95% CI.

    ``design`` must hold every sampled child, not only those in the domain,
    and must carry ``stratum_id`` and ``hhid``.
    """
    w = pl.col(weight)
    denominator = float(design.select((w * pl.col(valid)).sum()).item())
    numerator = float(design.select((w * pl.col(yes)).sum()).item())
    p = numerator / denominator
    # Linearized score for a ratio. Zero for children outside the domain.
    se = _linearized_se(design, w * pl.col(valid) * (pl.col(yes) - p) / denominator)
    # The logit interval keeps the bounds inside (0, 1) and is the type the
    # Census SAS examples request.
    logit = math.log(p / (1 - p))
    half = _Z_95 * se / (p * (1 - p))
    low = 1 / (1 + math.exp(-(logit - half)))
    high = 1 / (1 + math.exp(-(logit + half)))
    return Estimate(value=p, se=se, low=low, high=high)


def weighted_total(design: pl.DataFrame, yes: str, weight: str) -> Estimate:
    """Weighted population count of ``yes``, with a symmetric 95% CI."""
    w = pl.col(weight)
    total = float(design.select((w * pl.col(yes)).sum()).item())
    se = _linearized_se(design, w * pl.col(yes))
    return Estimate(value=total, se=se, low=total - _Z_95 * se, high=total + _Z_95 * se)


def prevalence_row(design: pl.DataFrame, weight: str) -> dict[str, float | int]:
    """Every reported quantity for one design frame (one year, or pooled years).

    ``design`` is the output of :func:`add_indicators` and must contain every
    sampled child for the years it covers.
    """
    row: dict[str, float | int] = {
        "children_in_age_band": int(design.select(pl.col("in_age").sum()).item()),
    }
    for outcome in ("ever", "current"):
        yes, valid = f"{outcome}_yes", f"{outcome}_valid"
        n_yes = int(design.select(pl.col(yes).sum()).item())
        n_valid = int(design.select(pl.col(valid).sum()).item())
        proportion = weighted_proportion(design, yes, valid, weight)
        total = weighted_total(design, yes, weight)
        row |= {
            f"{outcome}_n": n_yes,
            f"{outcome}_denominator": n_valid,
            f"{outcome}_pct_unweighted": 100 * n_yes / n_valid,
            f"{outcome}_pct_weighted": 100 * proportion.value,
            f"{outcome}_pct_low": 100 * proportion.low,
            f"{outcome}_pct_high": 100 * proportion.high,
            f"{outcome}_population": total.value,
            f"{outcome}_population_low": total.low,
            f"{outcome}_population_high": total.high,
        }
    return row


def prevalence_table(children: pl.DataFrame, years: Iterable[int]) -> pl.DataFrame:
    """One row per survey year plus a pooled ``"All years"`` row.

    The pooled row divides each weight by the number of years, so its
    population figure is an average per year, not a nine-year sum.
    """
    year_list = list(years)
    rows = [
        {"period": str(year)} | prevalence_row(children.filter(pl.col("year") == year), "fwc")
        for year in year_list
    ]
    pooled = children.with_columns(fwc_pooled=pl.col("fwc") / len(year_list))
    rows.append({"period": "All years"} | prevalence_row(pooled, "fwc_pooled"))
    return pl.DataFrame(rows)


def read_do_labels(path: Path | str, variable: str) -> dict[str, str]:
    """Pull one variable's label and value labels out of a Census ``.do`` file.

    The public-use ``.dta`` files carry no embedded value labels; the meaning
    of each code lives in the accompanying ``.do`` file. Reading it here lets
    a notebook show, in the Census Bureau's own words, that ``1`` means Yes in
    every year.

    Returns
    -------
    dict[str, str]
        ``{"variable_label": ..., "1": "Yes", "2": "No", ".m": ..., ...}``
    """
    text = Path(path).read_text(encoding="latin-1")
    out: dict[str, str] = {}
    var_match = re.search(rf'label var\s+{variable}\s+"([^"]*)"', text, flags=re.IGNORECASE)
    if var_match is None:
        raise KeyError(f"{Path(path).name} has no variable label for {variable}")
    out["variable_label"] = var_match.group(1).strip()
    pattern = rf'label define\s+{variable}_lab\s+(\S+)\s+"([^"]*)"'
    for code, label in re.findall(pattern, text, flags=re.IGNORECASE):
        out[code] = label.strip()
    return out
