import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import datetime
    import os
    import pathlib

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl

    from nsch_ml import prevalence as pv

    return datetime, mo, os, pathlib, pl, plt, pv


@app.cell
def _(mo):
    mo.md(r"""
    # Autism among children aged 3 to 17 in the NSCH, 2016 to 2024

    This notebook answers one question: how many children aged 3 to 17
    have **ever** had autism, and how many **currently** have it, in each
    National Survey of Children's Health from 2016 to 2024 and across all
    nine years together.

    I built it so that every figure can be traced back to the Census
    Bureau's public files. It reads those files directly, so nothing from
    the modelling pipeline sits between the source and the count. Every
    number on this page is computed when the notebook runs, with one
    labelled exception: a published CAHMI table in Check 1, typed in so our
    results have something outside this project to be compared against.

    The order is the files, then the two survey questions, then an
    accounting of every child, then the definitions and the results, and
    finally four checks that have to pass before anything above them is
    worth quoting.
    """)
    return


@app.cell
def _(mo, os, pathlib):
    # The raw Census files live outside the repository, and no committed file
    # names anyone's home directory. Refuse to run instead of guessing a path,
    # because a guessed path is a guessed dataset.
    _configured = os.environ.get("NSCH_RAW", "")
    raw_dir = pathlib.Path(_configured).expanduser() if _configured else None
    mo.stop(
        raw_dir is None or not raw_dir.is_dir(),
        mo.md(
            "**This notebook cannot run.** `NSCH_RAW` is not set, or does not point at a "
            "directory. It should hold the unzipped Census topical `.dta` and `.do` files "
            "for 2016 to 2024. See `notebooks/README.md`."
        ),
    )
    years = list(range(2016, 2025))
    min_age = 3
    max_age = 17
    return max_age, min_age, raw_dir, years


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The files

    One topical file per year, downloaded from the Census Bureau's
    [NSCH dataset listing](https://www2.census.gov/programs-surveys/nsch/datasets/).
    The SHA-256 fingerprint identifies the exact file. That matters here
    because the Bureau revised its weighting method with the 2022 survey
    and reissued the 2016 to 2021 files with the new weights, so two people
    who both hold "the 2016 file" can hold different files. Anyone rerunning
    this can compare fingerprints before comparing numbers.
    """)
    return


@app.cell
def _(datetime, mo, pl, pv, raw_dir, years):
    dta_paths = {year: pv.find_topical_file(raw_dir, year) for year in years}
    provenance = pl.DataFrame(
        [
            {
                "year": year,
                "file": path.name,
                "size_mb": round(path.stat().st_size / 1e6, 1),
                "file_modified": datetime.datetime.fromtimestamp(
                    path.stat().st_mtime, tz=datetime.UTC
                )
                .date()
                .isoformat(),
                "sha256": pv.sha256_of(path),
            }
            for year, path in dta_paths.items()
        ]
    )
    mo.ui.table(provenance, selection=None, pagination=False)
    return (dta_paths,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The two survey questions

    Two items carry the whole calculation. **K2Q35A** asks whether a doctor
    or other health care provider ever told the parent that the child has
    autism or autism spectrum disorder. **K2Q35B** is asked only when that
    answer is Yes, and asks whether the child currently has the condition.

    The table below is read from each year's Census `.do` file, which is
    where the meaning of every code lives. It shows that the codes mean the
    same thing in all nine years. The cell refuses to continue if any year
    differs. CAHMI's own crosswalk records no substantive change to either
    item since 2016, and the `.do` files agree.
    """)
    return


@app.cell
def _(dta_paths, mo, pl, pv, raw_dir):
    _rows = []
    for _year in dta_paths:
        _do_path = pv.find_topical_file(raw_dir, _year, suffix=".do")
        for _variable in ("k2q35a", "k2q35b"):
            _rows.append(
                {"year": _year, "variable": _variable} | pv.read_do_labels(_do_path, _variable)
            )
    do_labels = pl.DataFrame(_rows)
    _distinct = do_labels.drop("year").unique().sort("variable")
    mo.stop(
        _distinct.height != 2,
        mo.vstack(
            [
                mo.md(
                    "**Stopped.** The K2Q35A or K2Q35B labels differ between years. "
                    "The full table is below."
                ),
                mo.ui.table(do_labels, selection=None, pagination=False),
            ]
        ),
    )
    mo.vstack(
        [
            mo.md(f"One label set per variable across all {do_labels['year'].n_unique()} years."),
            mo.ui.table(_distinct, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Every child accounted for

    Each row is a survey year. Reading left to right: every child in the
    file, the children aged 3 to 17, and then those children split into
    five groups that cannot overlap. The five groups have to add up to the
    3 to 17 total in every year, and the cell stops if they do not.

    The read step also refuses if a file's own `year` column disagrees
    with the year expected of it, or if an answer pattern turns up that the
    five groups do not cover.
    """)
    return


@app.cell
def _(dta_paths, max_age, min_age, mo, pl, pv):
    _raw = pl.concat([pv.read_topical_year(path, year) for year, path in dta_paths.items()])
    children = pv.add_indicators(pv.classify_autism(_raw), min_age, max_age)

    _by_year = (
        children.group_by("year")
        .agg(
            all_children_in_file=pl.len(),
            aged_3_to_17=pl.col("in_age").sum(),
            **{
                group: ((pl.col("autism_group") == group) & (pl.col("in_age") == 1)).sum()
                for group in pv.GROUPS
            },
        )
        .sort("year")
    )
    accounting = pl.concat(
        [
            _by_year.with_columns(pl.col("year").cast(pl.String)),
            _by_year.drop("year").sum().select(pl.lit("All years").alias("year"), pl.all()),
        ],
        how="vertical_relaxed",
    )
    _group_sum = accounting.select(pl.sum_horizontal(list(pv.GROUPS))).to_series().to_list()
    mo.stop(
        _group_sum != accounting["aged_3_to_17"].to_list(),
        mo.md("**Stopped.** The five groups do not add up to the children aged 3 to 17."),
    )
    mo.ui.table(accounting, selection=None, pagination=False)
    return accounting, children


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Definitions

    | | Counted as yes | In the denominator |
    |---|---|---|
    | **Ever had autism** | K2Q35A is Yes | K2Q35A is Yes or No |
    | **Currently has autism** | K2Q35A is Yes and K2Q35B is Yes | K2Q35A is No, or K2Q35A is Yes with K2Q35B answered |

    Children with no usable answer are left out of the denominator. That
    is the rule CAHMI's Data Resource Center uses for its autism indicator,
    and Check 1 shows our counts land on theirs. Check 2 shows how little
    the answer could move if those children had been handled any other way.

    Two kinds of number appear in the results.

    **Sample counts and unweighted percents** describe the children who
    were surveyed. These are the figures to cite as *n*.

    **Weighted percents and population estimates** describe children in
    the United States. They use the design the Census Bureau prescribes in
    its [Guide to Multi-Year Estimates](https://www2.census.gov/programs-surveys/nsch/technical-documentation/methodology/NSCH-Guide-to-Multi-Year-Estimates.pdf):
    the child weight `FWC`, strata formed from state and `STRATUM` (with
    `2A` recoded to `2`), the household as the sampling unit, and Taylor
    series standard errors with logit confidence intervals. For all years
    together each weight is divided by nine, as the guide directs, so the
    pooled population estimate is an **average per year**. It is not a
    nine-year sum.
    """)
    return


@app.cell
def _(children, mo, pl, pv, years):
    # Design checks the variance calculation depends on. Any failure means the
    # standard errors below would be wrong, so this stops instead of warning.
    _distinct_psus = children.select(pl.struct("stratum_id", "hhid").n_unique()).item()
    _strata_sizes = children.group_by(["year", "stratum_id"]).len()
    _bad_weights = children.filter(pl.col("fwc").is_null() | (pl.col("fwc") <= 0)).height
    _checks = {
        "a household ID repeats within a stratum": _distinct_psus != children.height,
        "a household ID repeats across years": children["hhid"].n_unique() != children.height,
        "a weight is missing, zero or negative": _bad_weights > 0,
        "STRATUM holds a value other than 1 or 2": sorted(children["stratum"].unique().to_list())
        != ["1", "2"],
        "a stratum has a single household in some year": _strata_sizes["len"].min() < 2,
    }
    _failed = [name for name, bad in _checks.items() if bad]
    mo.stop(len(_failed) > 0, mo.md(f"**Stopped.** Design check failed: {_failed}"))

    results = pv.prevalence_table(children, years)
    return (results,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Results

    The first table is formatted for reading. The second holds the same
    figures as plain numbers with more precision, and has a download
    button.
    """)
    return


@app.cell
def _(mo, pl, results):
    # Plain f-strings on a ten-row table: this is formatting for a reader, not
    # computation, so there is nothing to gain from expressions here.
    def _pct_ci(row: dict[str, float], stem: str) -> str:
        return (
            f"{row[f'{stem}_pct_weighted']:.2f}% "
            f"({row[f'{stem}_pct_low']:.2f} to {row[f'{stem}_pct_high']:.2f})"
        )

    display_table = pl.DataFrame(
        [
            {
                "Survey year": row["period"],
                "Children aged 3-17 (n)": f"{row['children_in_age_band']:,}",
                "Ever autism (n)": f"{row['ever_n']:,}",
                "Ever, % of sample": f"{row['ever_pct_unweighted']:.2f}%",
                "Ever, weighted % (95% CI)": _pct_ci(row, "ever"),
                "Ever, est. U.S. children": f"{row['ever_population']:,.0f}",
                "Current autism (n)": f"{row['current_n']:,}",
                "Current, % of sample": f"{row['current_pct_unweighted']:.2f}%",
                "Current, weighted % (95% CI)": _pct_ci(row, "current"),
                "Current, est. U.S. children": f"{row['current_population']:,.0f}",
            }
            for row in results.iter_rows(named=True)
        ]
    )
    mo.ui.table(display_table, selection=None, pagination=False)
    return


@app.cell
def _(mo, results):
    _pooled = results.filter(results["period"] == "All years").row(0, named=True)
    _first = results.row(0, named=True)
    _last = results.filter(results["period"] != "All years").row(-1, named=True)
    mo.callout(
        mo.md(
            f"""
            **In one paragraph.** Across the nine surveys, {_pooled["ever_n"]:,}
            of {_pooled["ever_denominator"]:,} surveyed children aged 3 to 17
            had ever been diagnosed with autism, and {_pooled["current_n"]:,}
            of {_pooled["current_denominator"]:,} currently had it. Weighted
            to the population, that is {_pooled["ever_pct_weighted"]:.2f}%
            ever ({_pooled["ever_pct_low"]:.2f} to {_pooled["ever_pct_high"]:.2f})
            and {_pooled["current_pct_weighted"]:.2f}% currently
            ({_pooled["current_pct_low"]:.2f} to {_pooled["current_pct_high"]:.2f}),
            or about {_pooled["current_population"] / 1e6:.2f} million children
            with autism in an average year. The weighted percent currently
            diagnosed rose from {_first["current_pct_weighted"]:.2f}% in
            {_first["period"]} to {_last["current_pct_weighted"]:.2f}% in
            {_last["period"]}.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo, results):
    mo.vstack(
        [
            mo.ui.table(results, selection=None, pagination=False),
            mo.download(
                data=results.write_csv().encode(),
                filename="nsch_autism_prevalence_2016_2024.csv",
                label="Download the results as CSV",
            ),
        ]
    )
    return


@app.cell
def _(mo, plt, results):
    _yearly = results.filter(results["period"] != "All years")
    _pooled = results.filter(results["period"] == "All years")
    _x = [int(period) for period in _yearly["period"].to_list()]
    colour_ever = "#1b6ca8"
    colour_current = "#d1603d"

    fig_pct, _ax = plt.subplots(figsize=(8, 4.5))
    for _stem, _colour, _label, _shift in (
        ("ever", colour_ever, "Ever had autism", -0.08),
        ("current", colour_current, "Currently has autism", 0.08),
    ):
        _mid = _yearly[f"{_stem}_pct_weighted"].to_list()
        _low = _yearly[f"{_stem}_pct_low"].to_list()
        _high = _yearly[f"{_stem}_pct_high"].to_list()
        _ax.errorbar(
            [value + _shift for value in _x],
            _mid,
            yerr=[
                [mid - low for mid, low in zip(_mid, _low, strict=True)],
                [high - mid for mid, high in zip(_mid, _high, strict=True)],
            ],
            fmt="o-",
            color=_colour,
            capsize=3,
            label=_label,
        )
        _ax.axhline(
            _pooled[f"{_stem}_pct_weighted"].item(), color=_colour, linestyle=":", linewidth=1
        )
    _ax.set_xticks(_x)
    _ax.set_ylim(bottom=0)
    _ax.set_ylabel("Weighted percent of U.S. children aged 3 to 17")
    _ax.set_title(
        "Weighted autism prevalence has risen since 2020, on both definitions\n"
        "(bars: 95% confidence intervals; dotted lines: all years pooled)",
        fontsize=10,
    )
    _ax.legend(frameon=False, loc="upper left")
    _ax.spines[["top", "right"]].set_visible(False)
    mo.as_html(fig_pct)
    return colour_current, colour_ever


@app.cell
def _(colour_current, colour_ever, mo, plt, results):
    _yearly = results.filter(results["period"] != "All years")
    _x = [int(period) for period in _yearly["period"].to_list()]

    fig_pop, _ax = plt.subplots(figsize=(8, 4.5))
    for _stem, _colour, _label, _shift in (
        ("ever", colour_ever, "Ever had autism", -0.08),
        ("current", colour_current, "Currently has autism", 0.08),
    ):
        _mid = [value / 1e6 for value in _yearly[f"{_stem}_population"].to_list()]
        _low = [value / 1e6 for value in _yearly[f"{_stem}_population_low"].to_list()]
        _high = [value / 1e6 for value in _yearly[f"{_stem}_population_high"].to_list()]
        _ax.errorbar(
            [value + _shift for value in _x],
            _mid,
            yerr=[
                [mid - low for mid, low in zip(_mid, _low, strict=True)],
                [high - mid for mid, high in zip(_mid, _high, strict=True)],
            ],
            fmt="o-",
            color=_colour,
            capsize=3,
            label=_label,
        )
    _ax.set_xticks(_x)
    _ax.set_ylim(bottom=0)
    _ax.set_ylabel("Estimated U.S. children aged 3 to 17 (millions)")
    _ax.set_title(
        "The estimated number of children with autism, by survey year\n"
        "(weighted population estimates with 95% confidence intervals)",
        fontsize=10,
    )
    _ax.legend(frameon=False, loc="upper left")
    _ax.spines[["top", "right"]].set_visible(False)
    mo.as_html(fig_pop)
    return


@app.cell
def _(colour_current, colour_ever, mo, plt, results):
    _yearly = results.filter(results["period"] != "All years")
    _x = list(range(_yearly.height))
    _width = 0.4

    fig_n, _ax = plt.subplots(figsize=(8, 4.5))
    for _stem, _colour, _label, _offset in (
        ("ever", colour_ever, "Ever had autism", -_width / 2),
        ("current", colour_current, "Currently has autism", _width / 2),
    ):
        _bars = _ax.bar(
            [value + _offset for value in _x],
            _yearly[f"{_stem}_n"].to_list(),
            _width,
            color=_colour,
            label=_label,
        )
        _ax.bar_label(_bars, fmt="{:,.0f}", fontsize=7, padding=2)
    _ax.set_xticks(_x, _yearly["period"].to_list())
    _ax.set_ylabel("Surveyed children aged 3 to 17 (sample count)")
    _ax.set_title(
        "Sample counts by survey year. These track the size of each year's sample\n"
        "as well as prevalence, so read the trend from the weighted percents above.",
        fontsize=10,
    )
    _ax.legend(frameon=False, loc="upper left")
    _ax.spines[["top", "right"]].set_visible(False)
    mo.as_html(fig_n)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Checks

    ### Check 1. Do we reproduce a published table?

    CAHMI's Data Resource Center publishes this measure as
    [Indicator 2.8](https://nschdata.org/browse/survey/results?q=11961&r=1)
    for 2023 and 2024 combined. The figures in the `published` rows were
    typed in from that page on 21 September 2026 and are the only typed
    numbers in this notebook. The `computed` rows come from the same code
    that produced the results above, restricted to 2023 and 2024 with each
    weight halved.

    Sample counts and population estimates have to match exactly, and
    percents have to match at CAHMI's one decimal place. The cell stops
    otherwise.
    """)
    return


@app.cell
def _(children, mo, pl, pv):
    _two_year = children.filter(pl.col("year").is_in([2023, 2024])).with_columns(
        w=pl.col("fwc") / 2,
        never_yes=((pl.col("in_age") == 1) & (pl.col("autism_group") == pv.GROUP_NEVER)).cast(
            pl.Int8
        ),
        not_current_yes=(
            (pl.col("in_age") == 1) & (pl.col("autism_group") == pv.GROUP_NOT_CURRENT)
        ).cast(pl.Int8),
    )
    # Transcribed from the CAHMI page linked above. Not computed here.
    _published = {
        "never_yes": ("Does not have condition", 85910, 58466857, 95.5),
        "not_current_yes": ("Ever told, but does not currently have condition", 182, 124114, 0.2),
        "current_yes": ("Currently has condition", 4120, 2661609, 4.3),
    }
    _rows = []
    for _column, (_label, _n, _population, _percent) in _published.items():
        _share = pv.weighted_proportion(_two_year, _column, "current_valid", "w")
        _total = pv.weighted_total(_two_year, _column, "w")
        _rows.append(
            {
                "category": _label,
                "source": "published",
                "sample_count": _n,
                "population": _population,
                "percent": _percent,
            }
        )
        _rows.append(
            {
                "category": _label,
                "source": "computed",
                "sample_count": int(_two_year[_column].sum()),
                "population": round(_total.value),
                "percent": round(100 * _share.value, 1),
            }
        )
    benchmark = pl.DataFrame(_rows)
    _published_side = benchmark.filter(pl.col("source") == "published").drop("source")
    _computed_side = benchmark.filter(pl.col("source") == "computed").drop("source")
    _table = mo.ui.table(benchmark, selection=None, pagination=False)
    mo.stop(
        not _published_side.equals(_computed_side),
        mo.vstack([mo.md("**Stopped.** The computed figures do not match CAHMI."), _table]),
    )
    mo.vstack(
        [
            mo.md(
                "**Match.** All three categories agree on sample count, population "
                "estimate and percent."
            ),
            _table,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Check 2. How much could the unanswered cases matter?

    Some parents skipped one or both questions. This table takes the two
    most extreme views possible: every one of those children does not have
    autism (lowest), or every one does (highest). Percents here are
    unweighted, with all children aged 3 to 17 as the denominator. The
    reported figure has to sit between the two, and the gap shows how much
    room the skipped answers leave.
    """)
    return


@app.cell
def _(accounting, mo, pl, pv):
    _n = pl.col("aged_3_to_17")
    _ever_unknown = pl.col(pv.GROUP_EVER_UNKNOWN)
    _current_unknown = pl.col(pv.GROUP_EVER_UNKNOWN) + pl.col(pv.GROUP_CURRENT_UNKNOWN)
    _ever = (
        pl.col(pv.GROUP_CURRENT) + pl.col(pv.GROUP_NOT_CURRENT) + pl.col(pv.GROUP_CURRENT_UNKNOWN)
    )
    _current = pl.col(pv.GROUP_CURRENT)
    sensitivity = accounting.select(
        "year",
        unanswered_ever=_ever_unknown,
        ever_pct_lowest=(100 * _ever / _n).round(2),
        ever_pct_reported=(100 * _ever / (_n - _ever_unknown)).round(2),
        ever_pct_highest=(100 * (_ever + _ever_unknown) / _n).round(2),
        unanswered_current=_current_unknown,
        current_pct_lowest=(100 * _current / _n).round(2),
        current_pct_reported=(100 * _current / (_n - _current_unknown)).round(2),
        current_pct_highest=(100 * (_current + _current_unknown) / _n).round(2),
    )
    mo.ui.table(sensitivity, selection=None, pagination=False)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Check 3. Do the results and the accounting agree?

    The results in section 5 and the accounting in section 3 are computed
    by different code paths from the same children. Their counts have to
    be equal.
    """)
    return


@app.cell
def _(accounting, mo, pl, pv, results):
    _from_accounting = accounting.select(
        ever_n=pl.col(pv.GROUP_CURRENT)
        + pl.col(pv.GROUP_NOT_CURRENT)
        + pl.col(pv.GROUP_CURRENT_UNKNOWN),
        current_n=pl.col(pv.GROUP_CURRENT),
        children_in_age_band=pl.col("aged_3_to_17"),
    ).cast(pl.Int64)
    _from_results = results.select("ever_n", "current_n", "children_in_age_band").cast(pl.Int64)
    mo.stop(
        not _from_accounting.equals(_from_results),
        mo.md("**Stopped.** Sections 3 and 5 disagree."),
    )
    mo.md("**Match.** Every count in the results table equals the accounting table.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Check 4. Does the harmonized modelling dataset give the same counts?

    This check is optional and answers a different question: whether the
    dataset our models train on agrees with the raw Census files on these
    two items. Set `NSCH_CLEAN` to a Parquet file written from the
    harmonized pipeline output to run it. The file needs the columns
    `year`, `sc_age_years`, `k2q35a` and `k2q35b`, with `Yes` and `No` as
    the labels.
    """)
    return


@app.cell
def _(max_age, min_age, mo, os, pathlib, pl, results):
    _configured = os.environ.get("NSCH_CLEAN", "")
    mo.stop(
        _configured == "",
        mo.md("**Skipped.** `NSCH_CLEAN` is not set, so this check did not run."),
    )
    _clean = pl.read_parquet(pathlib.Path(_configured).expanduser())
    _needed = {"year", "sc_age_years", "k2q35a", "k2q35b"}
    mo.stop(
        not _needed.issubset(_clean.columns),
        mo.md(f"**Stopped.** `NSCH_CLEAN` lacks columns: {sorted(_needed - set(_clean.columns))}"),
    )
    _in_age = _clean.filter(pl.col("sc_age_years").cast(pl.Int64).is_between(min_age, max_age))
    _ever = pl.col("k2q35a").cast(pl.String) == "Yes"
    _current = _ever & (pl.col("k2q35b").cast(pl.String) == "Yes")
    harmonized = (
        _in_age.group_by(pl.col("year").cast(pl.Int64).cast(pl.String).alias("period"))
        .agg(children_in_age_band=pl.len(), ever_n=_ever.sum(), current_n=_current.sum())
        .sort("period")
        .cast({"children_in_age_band": pl.Int64, "ever_n": pl.Int64, "current_n": pl.Int64})
    )
    _raw_side = (
        results.filter(pl.col("period") != "All years")
        .select(harmonized.columns)
        .cast(harmonized.schema)
    )
    mo.stop(
        not harmonized.equals(_raw_side),
        mo.vstack(
            [
                mo.md("**Stopped.** The harmonized counts differ from the raw files."),
                mo.ui.table(harmonized, selection=None, pagination=False),
                mo.ui.table(_raw_side, selection=None, pagination=False),
            ]
        ),
    )
    mo.md(
        "**Match.** The harmonized dataset gives the same counts as the raw Census files "
        "in every year."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Reading these numbers

    All of this is parent report. A parent said a provider told them the
    child has autism. No clinical record stands behind it, and CAHMI's
    indicator page says the same.

    The percent of the sample and the weighted percent differ because the
    survey oversamples some households and the sample size changes from
    year to year. For a statement about children in the United States, use
    the weighted percent and its interval. For describing who is in our
    analytic sample, use the sample counts.

    The population estimate for all years together is an average per year.
    It is not a count of distinct children over nine years, and the yearly
    estimates should not be added up.

    The Census Bureau revised its weighting beginning with the 2022 survey
    and reissued the earlier files. The fingerprints in section 1 record
    which release produced these figures.

    Sources: Census Bureau
    [NSCH datasets](https://www.census.gov/programs-surveys/nsch/data/datasets.html)
    and
    [Guide to Multi-Year Estimates](https://www2.census.gov/programs-surveys/nsch/technical-documentation/methodology/NSCH-Guide-to-Multi-Year-Estimates.pdf);
    CAHMI Data Resource Center,
    [Indicator 2.8, 2023-2024](https://nschdata.org/browse/survey/results?q=11961&r=1).
    """)
    return


if __name__ == "__main__":
    app.run()
