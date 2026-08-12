import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from scipy import stats

    # This notebook lives in notebooks/; the analysis scripts are its siblings.
    # Importing them rather than copying their logic keeps one definition of
    # every quantity, so a correction to a script reaches this document too.
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "analyses"))

    import compare_published_registry as registry_tools
    import prediction_equivalence as equivalence
    import soak_ttests as ttests

    def resolve_input_path(variable, *parts):
        """Locate an input file from the environment, or say why it failed.

        Inputs live outside this repository, and no committed file should name
        anyone's home directory. Returns the path and None, or None and a
        message the notebook shows in place of its results.
        """
        configured = os.environ.get(variable)
        if not configured:
            return None, f"`{variable}` is not set. See notebooks/README.md."
        path = Path(configured).expanduser().joinpath(*parts)
        if not path.exists():
            return None, f"`{variable}` is set, but {path} does not exist."
        return path, None

    fixture_path, _fixture_problem = resolve_input_path("REPRO", "data_Classif", "NSCH_autism.csv")
    r_predictions_path, _r_problem = resolve_input_path(
        "REPRO", "results", "predictions", "NSCH_seed1_predictions_repaired.csv"
    )
    python_predictions_path, _python_problem = resolve_input_path(
        "REPRO", "results", "predictions", "python_lasso_seed1_is100_predictions.csv"
    )
    registry_path, _registry_problem = resolve_input_path(
        "PAPER", "data_Classif_batchmark_registry.csv"
    )
    service_use_path, _service_problem = resolve_input_path(
        "MONSOON_OLD", "2016_2023_ServiceUse.csv"
    )

    replication_results_path = repo_root / "analyses" / "glmnet_replication_lasso_seed1_is100.csv"
    featureless_results_path = repo_root / "analyses" / "featureless_replication.csv"

    missing_inputs = [
        problem
        for problem in (
            _fixture_problem,
            _r_problem,
            _python_problem,
            _registry_problem,
            _service_problem,
        )
        if problem
    ] + [
        f"{path} is missing."
        for path in (replication_results_path, featureless_results_path)
        if not path.exists()
    ]
    return (
        equivalence,
        featureless_results_path,
        fixture_path,
        missing_inputs,
        mo,
        np,
        pl,
        plt,
        python_predictions_path,
        r_predictions_path,
        registry_path,
        registry_tools,
        replication_results_path,
        service_use_path,
        stats,
        ttests,
    )


@app.cell
def _(mo, missing_inputs):
    mo.md("\n\n".join(["**This notebook cannot run.**", *missing_inputs]) if missing_inputs else "")
    return


@app.cell
def _():
    def markdown_table(headers, alignments, body_rows, indent=8):
        """Build a markdown table that survives being interpolated into mo.md.

        mo.md strips the common leading whitespace from whatever it is given.
        A table assembled by joining rows on a bare newline leaves those rows
        flush left, which drops the common prefix to zero; the surrounding
        prose then keeps its eight spaces and markdown renders it as a code
        block instead of text. Padding every generated line to the same depth
        as the docstring keeps the prefix uniform and the table intact.

        Alignments are 'left', 'right' or 'center', one per column.
        """
        rules = {"left": "---", "right": "---:", "center": ":---:"}
        divider = [rules[alignment] for alignment in alignments]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(divider) + " |",
            *body_rows,
        ]
        return ("\n" + " " * indent).join(lines)

    def spell_count(number):
        """Small counts read better as words in running prose."""
        words = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
        }
        return words.get(number, f"{number:,}")

    return markdown_table, spell_count


@app.cell
def _(mo):
    mo.md(
        r"""
        # Rebuilding the autism-prediction analysis in Python

        Chris Reger, ASD3 Outcomes Project. 12 August 2026.

        Every number and figure below is computed at the moment this page is
        built, from the result files named beside it. Nothing here is typed in
        by hand or carried over from an earlier draft.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. What you are looking at

        Vince, our previous analyst, built an analysis in R that predicts, from
        survey answers, which children have a reported autism diagnosis. It
        uses a cross-validation method published by Toby Hocking, who is one of
        our collaborators, and it is the foundation for the outcomes work in
        the manuscript.

        I want to extend that analysis to new questions about health care
        access. Before extending something, I would rather know it does what I
        think it does, and the honest way to find that out is to build it again
        from scratch and see whether the two versions agree. So I rebuilt it in
        Python, independently, and compared the results.

        **They agree.** Three independent computations of the same quantity
        land on top of each other. Both versions support the same scientific
        conclusion. And a deliberately trivial model reproduces the published
        result to six decimal places, which tells me the data underneath both
        versions is the same data.

        Two things turned up along the way that change what the manuscript can
        claim. The published analysis makes no use of the survey weights, so
        its results describe the children who happened to be surveyed rather
        than American children in general. And the two survey years are not the
        same size, which turns out to matter for how one of the comparisons
        should be read. Both are in sections 4 and 8.

        Sections 2 through 8 are the replication. Sections 9 through 11 are the
        new work, and section 9 has a question I need answered.
        """
    )
    return


@app.cell
def _(fixture_path, pl):
    survey_matrix = pl.scan_csv(fixture_path)
    _column_names = survey_matrix.collect_schema().names()

    # Two columns are not features: survey_year is the batch label the method
    # splits on, and y is the outcome being predicted. The outcome arrives as
    # text or as 0/1 depending on how the matrix was written, so read the
    # levels off the file rather than assuming one or the other.
    outcome_levels = (
        survey_matrix.select(pl.col("y").value_counts(sort=True))
        .collect()
        .unnest("y")
        .sort("count")
    )
    _positive_label = (
        "Yes" if "Yes" in [str(value) for value in outcome_levels["y"].to_list()] else 1
    )

    children_by_year = (
        survey_matrix.group_by("survey_year")
        .agg(
            pl.len().alias("children"),
            (pl.col("y") == _positive_label).sum().alias("with_diagnosis"),
        )
        .sort("survey_year")
        .collect()
        .with_columns((pl.col("with_diagnosis") / pl.col("children")).alias("share"))
    )
    feature_count = len(_column_names) - 2
    total_children = int(children_by_year["children"].sum())
    smallest_year_size = int(children_by_year["children"].min())
    largest_year_size = int(children_by_year["children"].max())
    return (
        children_by_year,
        feature_count,
        largest_year_size,
        smallest_year_size,
        total_children,
    )


@app.cell
def _(
    children_by_year,
    feature_count,
    largest_year_size,
    markdown_table,
    mo,
    smallest_year_size,
    spell_count,
    total_children,
):
    _year_rows = [
        f"| {row['survey_year']} | {row['children']:,} | {row['with_diagnosis']:,} "
        f"| {row['share']:.1%} |"
        for row in children_by_year.iter_rows(named=True)
    ]
    _table = markdown_table(
        ["Survey year", "Children", "With a reported diagnosis", "Share"],
        ["left", "right", "right", "right"],
        _year_rows,
    )
    _overall_share = int(children_by_year["with_diagnosis"].sum()) / total_children
    _size_ratio = largest_year_size / smallest_year_size
    mo.md(
        f"""
        ## 2. The data, in one card

        The National Survey of Children's Health is an annual survey of
        American families, covering health, health care, and family life. A
        parent or guardian answers on behalf of one child in the household.

        This analysis uses **{total_children:,} children** across
        **{spell_count(children_by_year.height)} survey years**, each described
        by **{feature_count} survey answers**. The thing being predicted is
        whether the parent reported an autism diagnosis.

        {_table}

        Two features of this table matter later, so they are worth pausing on.

        **About {_overall_share:.0%} of these children have a reported
        diagnosis.** When roughly three children in a hundred have the thing
        you are trying to find, a model that simply answers "no" to every child
        is correct about 97% of the time. It has learned nothing, and it would
        still post an accuracy score most people would call excellent. Accuracy
        is close to useless as a measure here.

        The measure used instead is **AUC**. Picture drawing two children at
        random, one who has a diagnosis and one who does not. AUC is the
        probability that the model gives the first child the higher score. A
        model guessing at random scores 0.5. A model that ranks every child
        with a diagnosis above every child without one scores 1.0. Because it
        only asks about the ordering of pairs, it is unaffected by how rare the
        outcome is, which is exactly the property accuracy lacks.

        **The two years are not the same size.** One holds
        {smallest_year_size:,} children and the other {largest_year_size:,},
        a ratio of about {_size_ratio:.1f} to 1. That imbalance has no bearing
        on whether the Python rebuild matches the R original, which is what
        sections 4 through 7 are about. It has a great deal of bearing on how
        section 6's comparisons should be read, and I come back to it there.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. The question the method is built to answer

        A practical problem turns up whenever data arrives in batches, and
        survey years are batches. Suppose you have two years of responses. Do
        you pool them and fit one model, or does each year need its own?

        Pooling gives you more children to learn from, which usually helps. But
        it only helps if the years are alike. If something changed between them
        in the way questions were asked, or in who responded, or in the world
        the children live in, then pooling means training partly on
        circumstances that no longer apply. The tradeoff is real and it is not
        obvious in advance which way it falls.

        SOAK, the method Toby developed, settles the question empirically
        instead of by argument. It holds out a portion of one year's children,
        then trains three models and asks which one predicts those held-out
        children best.

        - **Same**: train only on other children from the same survey year.
        - **Other**: train only on children from the different survey year.
        - **All**: train on both.

        Because all three are judged on the very same held-out children, the
        comparison between them is like for like. Each is repeated ten times
        over different held-out portions, for each of the two years, which
        gives sixty train-and-test runs.

        Two comparisons carry the meaning. **All against Same** asks whether
        pooling helps. **Other against Same** asks how well one year transfers
        to the other.
        """
    )
    return


@app.cell
def _(np, plt):
    def draw_soak_schematic():
        """A picture of what the three training sets contain, for section 3."""
        figure, axis = plt.subplots(figsize=(8.6, 2.9))
        year_colors = {"2019": "#4C6E9C", "2020": "#9C7A4C"}
        strategies = [
            ("Test on 2019", ["2019"], "one held-out tenth of 2019"),
            ("Same", ["2019"], "the other nine tenths of 2019"),
            ("Other", ["2020"], "all of 2020"),
            ("All", ["2019", "2020"], "both, minus the held-out tenth"),
        ]
        for position, strategy in enumerate(strategies):
            years, description = strategy[1], strategy[2]
            row = len(strategies) - position - 1
            left_edge = 0.0
            for year in years:
                # The held-out row is drawn faint: it is the target of the
                # comparison, not one of the three things being compared.
                axis.barh(
                    row,
                    1.0,
                    left=left_edge,
                    height=0.55,
                    color=year_colors[year],
                    alpha=0.35 if position == 0 else 0.85,
                    edgecolor="white",
                )
                axis.text(
                    left_edge + 0.5,
                    row,
                    year,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#20304A" if position == 0 else "white",
                )
                left_edge += 1.0
            axis.text(2.15, row, description, va="center", fontsize=9, color="#333333")
        axis.set_yticks(np.arange(len(strategies))[::-1])
        axis.set_yticklabels([strategy[0] for strategy in strategies], fontsize=10)
        axis.set_xlim(0, 5.6)
        axis.set_xticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title(
            "Three training sets, compared against the same held-out children",
            fontsize=11,
            loc="left",
        )
        figure.tight_layout()
        return figure

    draw_soak_schematic()
    return


@app.cell
def _(pl, registry_path, registry_tools, replication_results_path):
    # The single Python run that every figure in docs/replication-equivalence.md
    # comes from. Mixing runs is how tables end up quietly inconsistent, so this
    # notebook uses one and only one. Downsampled variants are dropped: this
    # analysis trains on full-size training sets throughout.
    replication_splits = (
        pl.read_csv(replication_results_path)
        .filter(~pl.col("downsampled") & pl.col("r_auc").is_not_null())
        .with_columns(pl.col("test_subset").cast(pl.Utf8))
    )

    # The results the SOAK paper was published from, filtered to this task and
    # learner by the same loader the comparison script uses.
    published_splits = registry_tools.load_registry(registry_path)

    test_years = sorted(replication_splits["test_subset"].unique().to_list())
    return published_splits, replication_splits, test_years


@app.cell
def _(published_splits, ttests):
    published_color = "#5C5C5C"

    implementation_series = (
        ("Published", published_color),
        ("R", ttests.R_COLOR),
        ("Python", ttests.PYTHON_COLOR),
    )

    def fold_auc_values(splits, implementation, test_year, train_source):
        """The ten per-fold AUCs for one cell, from one of the three sources.

        The published run and our two runs live in different files under
        different column names, so this is the single place that knows which
        column belongs to which implementation.
        """
        if implementation == "Published":
            return (
                published_splits.filter(
                    (published_splits["test_subset"] == test_year)
                    & (published_splits["train_source"] == train_source)
                )
                .sort("fold")["auc"]
                .to_numpy()
                .astype(float)
            )
        column = "r_auc" if implementation == "R" else "auc_1se"
        return ttests.fold_series(splits, test_year, train_source, column)

    return fold_auc_values, implementation_series


@app.cell
def _(fold_auc_values, implementation_series, np, plt, replication_splits, test_years, ttests):
    def draw_headline_comparison():
        """Every fold from every implementation, in the SOAK paper's own idiom."""
        figure, axes = plt.subplots(
            1, len(test_years), figsize=(5.6 * len(test_years), 4.0), sharex=True, sharey=True
        )
        axes = np.atleast_1d(axes)
        for axis, test_year in zip(axes, test_years, strict=True):
            for source_position, train_source in enumerate(ttests.TRAIN_SOURCES):
                for series_position, (implementation, color) in enumerate(implementation_series):
                    values = fold_auc_values(
                        replication_splits, implementation, test_year, train_source
                    )
                    if len(values) == 0:
                        continue
                    # Nudge the three implementations apart vertically so their
                    # circles stay legible where the values coincide.
                    row = source_position + (series_position - 1) * 0.24
                    axis.scatter(
                        values,
                        np.full(len(values), row),
                        facecolors="none",
                        edgecolors=color,
                        s=26,
                        linewidths=0.9,
                        alpha=0.75,
                        zorder=2,
                    )
                    axis.errorbar(
                        [values.mean()],
                        [row],
                        xerr=[values.std(ddof=1)],
                        fmt="o",
                        color=color,
                        markersize=6,
                        capsize=3,
                        linewidth=1.6,
                        zorder=3,
                        label=(
                            implementation if (axis is axes[0] and source_position == 0) else None
                        ),
                    )
            axis.set_yticks(np.arange(len(ttests.TRAIN_SOURCES)))
            axis.set_yticklabels(
                [source.capitalize() for source in ttests.TRAIN_SOURCES], fontsize=10
            )
            axis.set_title(f"Tested on {test_year}", fontsize=10)
            axis.set_xlabel("AUC. One circle per split; filled marker is the mean, bar is ±1 SD")
            axis.grid(axis="x", alpha=0.25, linewidth=0.6)
            axis.invert_yaxis()
        axes[0].legend(frameon=False, fontsize=9, loc="lower left")
        figure.suptitle(
            "Three independent computations of the same analysis land in the same place",
            fontsize=12,
        )
        figure.tight_layout()
        return figure

    draw_headline_comparison()
    return


@app.cell
def _(
    fold_auc_values,
    implementation_series,
    markdown_table,
    mo,
    replication_splits,
    test_years,
    ttests,
):
    _headline_rows = []
    _signed_gaps = []
    for _test_year in test_years:
        for _train_source in ttests.TRAIN_SOURCES:
            _means = {
                implementation: fold_auc_values(
                    replication_splits, implementation, _test_year, _train_source
                ).mean()
                for implementation, _ in implementation_series
            }
            _gap = _means["Python"] - _means["R"]
            _signed_gaps.append(_gap)
            _headline_rows.append(
                f"| {_test_year} | {_train_source.capitalize()} "
                f"| {_means['Published']:.4f} | {_means['R']:.4f} | {_means['Python']:.4f} "
                f"| {_gap:+.4f} |"
            )
    _table = markdown_table(
        ["Tested on", "Trained on", "Published", "R", "Python", "Python minus R"],
        ["left", "left", "right", "right", "right", "right"],
        _headline_rows,
    )
    _largest_gap = max(abs(gap) for gap in _signed_gaps)
    _all_same_sign = all(gap > 0 for gap in _signed_gaps) or all(gap < 0 for gap in _signed_gaps)
    _mean_gap = sum(_signed_gaps) / len(_signed_gaps)
    mo.md(
        f"""
        ## 4. The headline comparison

        Each row of the figure above is one cell of the design: a test year
        paired with a training strategy. Within a row there are thirty circles,
        ten folds from each of three sources, and the filled marker is that
        source's average.

        The table gives the same thing as numbers. **Published** is what
        appears in Toby's paper, **R** is our own rerun of his code, and
        **Python** is my rebuild.

        {_table}

        The largest disagreement between R and Python anywhere in the table is
        **{_largest_gap:.4f}**, on a scale where the measure itself runs from
        0.5 to 1.

        The comparison I would actually draw your attention to is not that
        number, though. Look at how far the ten circles in any single row
        spread out. That spread is the ordinary variation you get from
        splitting the same data a different way, and it is visibly wider than
        the distance between the three coloured markers. The choice of
        programming language moves the result less than the luck of which
        children happened to land in which fold.

        **Two caveats, both of which a careful reader will spot before I say
        them.**

        The first is that the Python column is
        {"above" if _mean_gap > 0 else "below"} the R column in
        {"every" if _all_same_sign else "most"} cell, by about
        {abs(_mean_gap):.4f} on average. A difference that never changes sign
        is not random noise; it is a small systematic offset, and I want to be
        clear that I have characterised it rather than explained it. I do not
        currently know its cause. What I can say is that it cancels out of
        every comparison in section 6, because those comparisons are
        differences between two cells and the offset sits in both.

        The second is that the published run used an earlier version of the
        resampling code, which numbered its folds differently from ours. Its
        ten circles are therefore not matched to our ten one for one, and only
        its average is comparable. That is why the rightmost column compares
        Python to R rather than to the published figures.
        """
    )
    return


@app.cell
def _(equivalence, np, pl, python_predictions_path, r_predictions_path, stats):
    # Both files carry one row per held-out child per split, so joining on the
    # split identity plus the child's row id lines up the same child's two
    # estimates. py_prob_1se is the like-for-like column: R's cv_glmnet
    # predicts at lambda.1se, and this is its counterpart.
    matched_predictions = (
        pl.read_csv(r_predictions_path)
        .with_columns(pl.col("test_subset").cast(pl.Utf8))
        .join(
            pl.read_csv(python_predictions_path)
            .with_columns(pl.col("test_subset").cast(pl.Utf8))
            .select("test_subset", "train_source", "fold", "row_id", "py_prob_1se"),
            on=["test_subset", "train_source", "fold", "row_id"],
            how="inner",
        )
    )

    _rank_correlations = []
    for _split, _predictions in matched_predictions.group_by(
        ["test_subset", "train_source", "fold"], maintain_order=True
    ):
        _rank_correlations.append(
            stats.spearmanr(
                _predictions["r_prob"].to_numpy(), _predictions["py_prob_1se"].to_numpy()
            ).statistic
        )
    rank_correlations = np.asarray(_rank_correlations)
    splits_meeting_standard = int((rank_correlations >= equivalence.SPEARMAN_FLOOR).sum())
    mean_probability_gap = float(
        np.abs(
            matched_predictions["r_prob"].to_numpy() - matched_predictions["py_prob_1se"].to_numpy()
        ).mean()
    )
    matched_row_count = matched_predictions.height
    return (
        matched_predictions,
        matched_row_count,
        mean_probability_gap,
        rank_correlations,
        splits_meeting_standard,
    )


@app.cell
def _(matched_predictions, plt):
    def draw_prediction_agreement():
        """Every held-out child, R's estimate against Python's."""
        figure, axis = plt.subplots(figsize=(5.8, 5.4))
        r_estimates = matched_predictions["r_prob"].to_numpy()
        python_estimates = matched_predictions["py_prob_1se"].to_numpy()
        # Counts are on a log scale because at a three percent base rate the
        # overwhelming majority of children sit near zero; on a linear scale
        # that one corner would be the only thing visible.
        density = axis.hexbin(
            r_estimates, python_estimates, gridsize=60, bins="log", cmap="Blues", mincnt=1
        )
        axis.plot([0, 1], [0, 1], color="#B85042", linewidth=1.1, linestyle="--", zorder=3)
        axis.set_xlabel("Probability estimated by the R model")
        axis.set_ylabel("Probability estimated by the Python model")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect("equal")
        figure.colorbar(density, ax=axis, label="children per cell (log scale)", shrink=0.82)
        axis.set_title(
            "Perfect agreement would put every child on the dashed line",
            fontsize=11,
        )
        figure.tight_layout()
        return figure

    draw_prediction_agreement()
    return


@app.cell
def _(
    equivalence,
    matched_row_count,
    mean_probability_gap,
    mo,
    rank_correlations,
    splits_meeting_standard,
):
    mo.md(
        f"""
        ## 5. Agreement child by child

        Averages can match while the things underneath them differ. Two models
        could produce the same average AUC by getting different children right,
        so section 4 on its own is not enough. The stricter test compares the
        two versions one child at a time.

        Across all **{matched_row_count:,}** held-out predictions, the average
        difference between the R estimate and the Python estimate for the same
        child is **{mean_probability_gap:.4f}**, or about a third of a
        percentage point.

        I judged this against a standard I wrote down and committed *before*
        running the comparison, in `docs/equivalence-margin.md`. Doing it in
        that order matters: had I looked at the results first, I could have
        picked whatever threshold my results happened to clear and called it a
        success. The standard asks that within each of the sixty runs, the two
        versions put children in the same order, at a rank correlation of at
        least {equivalence.SPEARMAN_FLOOR}.

        **{splits_meeting_standard} of {len(rank_correlations)}** runs meet it.
        The lowest correlation anywhere is {rank_correlations.min():.4f}.

        The {len(rank_correlations) - splits_meeting_standard} that fall short
        deserve a straight answer rather than a footnote. Both come from a
        single fitted model, and the disagreement is not about which children
        look likely to have a diagnosis. It is about the order of children the
        model is confident do *not*. At a three percent rate, the great
        majority of children sit in that confident-no group with nearly
        identical scores, so a tiny numerical difference reshuffles a long tail
        of near-ties and drags the rank correlation down while changing nothing
        about who the model actually flags.

        Whether a rank correlation is the right instrument for an outcome this
        rare is a fair question, and my own view is that it is a demanding one
        rather than a well-aimed one. But I chose it before seeing the results,
        so it stands. Loosening it now to bring two runs inside the line would
        turn the standard into a description of the answer rather than a test
        of it.
        """
    )
    return


@app.cell
def _(np, plt, replication_splits, test_years, ttests):
    auc_metric = ttests.build_metrics("auc_1se", "acc_1se")["auc"]

    def draw_soak_contrasts(metric):
        """The two SOAK comparisons, computed once in R and once in Python."""
        figure, axes = plt.subplots(1, len(ttests.CONTRASTS), figsize=(11.5, 3.9), sharex=True)
        axes = np.atleast_1d(axes)
        for axis, contrast_definition in zip(axes, ttests.CONTRASTS, strict=True):
            contrast_name, first_source, second_source = contrast_definition
            row_labels, row_positions, row = [], [], 0.0
            for test_year in test_years:
                for implementation, color in (
                    ("R", ttests.R_COLOR),
                    ("Python", ttests.PYTHON_COLOR),
                ):
                    contrast = ttests.paired_test(
                        ttests.side_series(
                            replication_splits, test_year, first_source, metric, implementation
                        ),
                        ttests.side_series(
                            replication_splits, test_year, second_source, metric, implementation
                        ),
                    )
                    axis.errorbar(
                        [contrast["mean"]],
                        [row],
                        xerr=[
                            [contrast["mean"] - contrast["lo"]],
                            [contrast["hi"] - contrast["mean"]],
                        ],
                        fmt="o",
                        color=color,
                        capsize=3,
                        markersize=5,
                        linewidth=1.4,
                    )
                    marker = " *" if contrast["p"] < ttests.ALPHA else ""
                    axis.annotate(
                        f"p={contrast['p']:.3f}{marker}",
                        (contrast["hi"], row),
                        textcoords="offset points",
                        xytext=(6, 0),
                        va="center",
                        fontsize=8,
                        color=color,
                    )
                    row_labels.append(f"{test_year}  {implementation}")
                    row_positions.append(row)
                    row += 1
                row += 0.6
            axis.axvline(0, color="#777777", linewidth=1, linestyle="--")
            axis.set_yticks(row_positions)
            axis.set_yticklabels(row_labels, fontsize=9)
            axis.set_title(contrast_name, fontsize=10)
            axis.set_xlabel("difference in AUC; positive favours the first named")
            axis.grid(axis="x", alpha=0.25, linewidth=0.6)
            axis.invert_yaxis()
            # Room on the right for the p-value annotations.
            axis.margins(x=0.3)
        figure.suptitle(
            "Pooling the survey years helps, and both versions agree that it does",
            fontsize=12,
        )
        figure.tight_layout()
        return figure

    draw_soak_contrasts(auc_metric)
    return (auc_metric,)


@app.cell
def _(
    auc_metric,
    largest_year_size,
    markdown_table,
    mo,
    replication_splits,
    smallest_year_size,
    spell_count,
    test_years,
    ttests,
):
    _contrast_rows = []
    _verdicts_agreeing = 0
    _verdicts_total = 0
    _disagreements = []
    for _test_year in test_years:
        for _contrast_name, _first, _second in ttests.CONTRASTS:
            _sides = {
                _implementation: ttests.paired_test(
                    ttests.side_series(
                        replication_splits, _test_year, _first, auc_metric, _implementation
                    ),
                    ttests.side_series(
                        replication_splits, _test_year, _second, auc_metric, _implementation
                    ),
                )
                for _implementation in ("R", "Python")
            }
            _calls = {
                implementation: ttests.significance_call(
                    result["mean"], result["p"], auc_metric.higher_is_better
                )
                for implementation, result in _sides.items()
            }
            _verdicts_total += 1
            if _calls["R"] == _calls["Python"]:
                _verdicts_agreeing += 1
            else:
                _disagreements.append(f"{_test_year} {_contrast_name}")
            _contrast_rows.append(
                f"| {_test_year} | {_contrast_name} | {_sides['R']['mean']:+.4f} "
                f"| {_sides['R']['p']:.4f} | {_sides['Python']['mean']:+.4f} "
                f"| {_sides['Python']['p']:.4f} |"
            )
    _table = markdown_table(
        [
            "Tested on",
            "Comparison",
            "R difference",
            "R p-value",
            "Python difference",
            "Python p-value",
        ],
        ["left", "left", "right", "right", "right", "right"],
        _contrast_rows,
    )
    _verdict_sentence = (
        f"the two implementations return the same verdict on all {_verdicts_total} comparisons"
        if not _disagreements
        else (
            f"the two implementations return the same verdict on {_verdicts_agreeing} of "
            f"{_verdicts_total} comparisons. The exception is "
            + " and ".join(_disagreements)
            + ", where one calls the difference significant and the other does not"
        )
    )
    mo.md(
        f"""
        ## 6. The science is unchanged

        Sections 4 and 5 show the two versions produce the same numbers. This
        section asks the question that actually matters: do they support the
        same conclusion? A replication that matched to four decimal places but
        pointed a reader somewhere else would be a failed one.

        A positive difference below means the first-named training set
        predicted better.

        {_table}

        **Pooling the years helps.** All beats Same in both test years, both
        implementations agree, and all four of those p-values are small. This
        is the substantive result of the original analysis, and it survives
        being rebuilt in another language. For the manuscript it is the
        evidence that combining survey years is defensible rather than merely
        convenient, which is what makes the multi-year extension in section 9
        worth doing.

        **The Other against Same comparison is a different story, and it is
        the one I would not over-read.** Its direction flips between the two
        years: training on the other year alone is slightly better when 2019 is
        held out, and slightly worse when 2020 is. That looks like a finding
        about how similar the years are. I think it is mostly a finding about
        how big they are. Recall from section 2 that the years hold
        {smallest_year_size:,} and {largest_year_size:,} children. When 2019 is
        held out, "Other" means the larger year and "Same" means the smaller
        one; when 2020 is held out, it is the reverse. A comparison that
        changes which side gets more training data is not cleanly measuring
        year-to-year transferability, and the sign flip is what you would
        predict from the sizes alone.

        I would not put weight on Other against Same in the manuscript without
        an analysis that holds training-set size fixed. The method supports one,
        by training on equal-sized subsets, and it is worth running before we
        say anything about how the years differ.

        On the replication question itself, {_verdict_sentence}.

        That single disagreement is worth dwelling on rather than glossing,
        because it is the honest illustration of the point above. Nothing about
        the underlying estimates is in dispute: both implementations put that
        difference in the same direction and at almost the same size. What
        differs is which side of 0.05 the p-value falls on, and a conclusion
        that can be flipped by an implementation detail this small was never
        resting on much to begin with. Reading the direction and the magnitude
        across all four comparisons is more informative than reading any one
        significance call.

        ### Three things a statistician would ask

        Included because they are fair questions and I would rather answer them
        here than be asked them later. Skip this block if it is not useful to
        you.

        *Are the sixty runs sixty independent observations?* No, and they are
        not treated as such. Within a given year and fold, all three training
        sources are scored on exactly the same held-out children, so their
        results move together. Each test is a paired t-test across the ten
        folds within one year, on nine degrees of freedom, which is the unit
        that is genuinely independent here.

        *Is there a multiplicity problem?* There are eight tests in the table,
        four comparisons computed twice, with no correction applied. I have
        reported all of them rather than the interesting ones, which is the
        honest version of not correcting. It is another reason to read the
        pattern across rows rather than any single p-value.

        *Are these differences large enough to matter?* In absolute terms they
        are small, a few thousandths of an AUC point. The claim being made is
        not that pooling transforms the model. It is that pooling does not hurt
        it, which is the question that has to be settled before the years can
        be combined at all.
        """
    )
    return


@app.cell
def _(featureless_results_path, markdown_table, mo, pl, registry_path):
    _ours = (
        pl.read_csv(featureless_results_path)
        .with_columns(pl.col("test_subset").cast(pl.Utf8))
        .group_by(["test_subset", "train_source"])
        .agg(pl.col("accuracy").mean().alias("ours"))
    )
    _published = (
        pl.read_csv(registry_path, infer_schema_length=20000)
        .filter(
            (pl.col("task_id") == "NSCH_autism") & pl.col("learner_id").str.contains("featureless")
        )
        .select(
            pl.col("test.group").cast(pl.Utf8).alias("test_subset"),
            pl.col("train.groups").alias("train_source"),
            # The registry records classification error; accuracy is its complement.
            (1.0 - pl.col("classif.ce")).alias("published"),
        )
        .group_by(["test_subset", "train_source"])
        .agg(pl.col("published").mean())
    )
    _both = _ours.join(_published, on=["test_subset", "train_source"], how="inner").sort(
        ["test_subset", "train_source"]
    )
    _baseline_rows = [
        f"| {row['test_subset']} | {row['train_source'].capitalize()} "
        f"| {row['published']:.6f} | {row['ours']:.6f} "
        f"| {abs(row['ours'] - row['published']):.6f} |"
        for row in _both.iter_rows(named=True)
    ]
    _table = markdown_table(
        ["Tested on", "Trained on", "Published", "Ours", "Difference"],
        ["left", "left", "right", "right", "right"],
        _baseline_rows,
    )
    _worst_gap = (
        float((_both["ours"] - _both["published"]).abs().max()) if _both.height else float("nan")
    )
    mo.md(
        f"""
        ## 7. The check that matched exactly

        Alongside the real model, the analysis runs a deliberately useless one.
        It ignores every survey answer and predicts the same thing for every
        child: the rate of diagnosis it saw in its training set.

        That sounds like a waste of a computation, and as a model it is. As a
        check it is the most informative thing in this document. Because it has
        no moving parts, its result depends on nothing except which children
        ended up in the training set and how many of them had a diagnosis. Any
        difference between two implementations would have to come from the data
        or the splits, because there is nothing else left for it to come from.

        {_table}

        The largest disagreement anywhere is **{_worst_gap:.6f}**.

        Matching to six zeros confirms three separate things at once: I loaded
        the same children, I coded the outcome the same way, and I divided them
        into the same groups as the published analysis. Whatever might still
        differ between the two versions in the modelling, the foundation
        underneath them is identical.

        This is also the one check here that is untouched by the fold-numbering
        caveat in section 4, because it does not depend on which fold is called
        which. And it is the reason I am comfortable attributing the small
        offset in section 4 to something in the modelling rather than to a
        difference in the data.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 8. Two things worth knowing before we go further

        **The published analysis makes no use of the survey weights.**

        The survey does not sample American children evenly. Some groups are
        deliberately oversampled so there are enough of them to say anything
        about, which means the raw sample is not a miniature of the country. To
        correct for this, the survey assigns every child a weight recording
        roughly how many children like them they stand for. Analyses that want
        to describe the national population have to use those weights.

        This model was offered the weight as one of its inputs and made no use
        of it. That is not a mistake in the original work and it is not a
        criticism. For a study asking which survey answers predict a diagnosis,
        the weights are beside the point.

        It matters for us because our manuscript wants to make claims about
        children in the United States, not about the particular children who
        answered this survey. Those are different claims and they need
        different machinery: survey-weighted estimation with standard errors
        that account for the survey's design. Building that version is the next
        deliverable, and this is the reason it is on the schedule.

        **The standard came before the result.**

        I mentioned in section 5 that the comparison threshold was committed
        before the comparison ran. Here is the concrete reason that discipline
        was worth the trouble.

        Two of our own R runs, differing in nothing but a random seed, disagree
        with each other by as much as 0.02 AUC on individual splits. That is
        substantially larger than any gap between R and Python in this
        document. So "the two runs are close" is not by itself a meaningful
        claim; it depends entirely on what counts as close, and that is a
        number I could have chosen after the fact to suit whatever I found.
        Fixing it in advance is what makes sections 4 and 5 a test rather than
        a description.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 9. What I am extending this to

        Everything above is groundwork. Predicting who has an autism diagnosis
        is not itself a useful thing to do, because the survey already records
        the diagnosis. The point was to establish that the machinery works, so
        it can be pointed at questions we do not already know the answer to.

        The real question is **which children with autism are not getting the
        care they need**, and whether that has been changing.

        Three outcomes, all from the same survey:

        | Outcome | A child counts as a case when | All children | Children with autism |
        | --- | --- | :---: | :---: |
        | Foregone care | there was a time in the past year they needed health care and did not receive it | yes | yes |
        | Emergency department use | they visited an emergency room in the past year | yes | yes |
        | Behaviour therapy | they received behavioural treatment for autism | no, the question is only asked about children who have a diagnosis | yes |

        **Two populations, two different questions.** Asking which children go
        without needed care, across all children, is a question about the
        health system in general. Asking which children *with autism* go
        without needed care is a question about how that system serves this
        group in particular. The denominators are different and so are the
        answers, so every table below states which population it describes.

        **One definition needs your call.** Emergency room visits are recorded
        in bands rather than as a count, which leaves us a choice. A case can
        mean one or more visits, which captures any contact with emergency care
        at all. Or it can mean two or more.

        My reason for raising the second option is thin and I want to be upfront
        about that. From a brief look at the literature, my impression is that
        repeat emergency visits are sometimes used as an indicator that
        routine care is not meeting a family's needs, on the reasoning that a
        single visit is often an unavoidable accident. I have not read enough
        to know how well established that is, whether it holds for children
        specifically, or whether it is the convention in the autism services
        literature you know far better than I do. Please correct me if the
        framing is off.

        Either way both are runnable, and section 10 shows what each looks like
        before any modelling. Tell me whether the manuscript wants one, the
        other, or both.
        """
    )
    return


@app.cell
def _(pl, service_use_path):
    # Emergency room visits arrive as a three-level variable, and the matrix
    # encodes the first two levels as indicator columns. A child with two or
    # more visits is therefore one whose indicators are both zero.
    any_emergency_visit = pl.col("hospitaler=None") == 0
    repeat_emergency_visits = any_emergency_visit & (pl.col("hospitaler=1 time") == 0)

    autism_subset_prevalence = (
        pl.read_csv(service_use_path, infer_schema_length=None)
        .group_by("period")
        .agg(
            pl.len().alias("children"),
            (pl.col("k4q27=Yes") == 1).sum().alias("Foregone care"),
            any_emergency_visit.sum().alias("ED use, one or more visits"),
            repeat_emergency_visits.sum().alias("ED use, two or more visits"),
            (pl.col("autismtreat=Yes") == 1).sum().alias("Behaviour therapy"),
        )
        .sort("period")
    )
    outcome_names = [
        "Foregone care",
        "ED use, one or more visits",
        "ED use, two or more visits",
        "Behaviour therapy",
    ]
    period_labels = {1: "2016-17", 2: "2018-19", 3: "2020-21", 4: "2022-23"}
    return autism_subset_prevalence, outcome_names, period_labels


@app.cell
def _(fixture_path, pl):
    # The same two outcomes on the full-population matrix, where the columns
    # carry descriptive names rather than survey codes.
    _any_emergency_visit = pl.col("Hospital_Emergency_Room_Visits_None") == 0
    _repeat_emergency_visits = _any_emergency_visit & (
        pl.col("Hospital_Emergency_Room_Visits_1_time") == 0
    )

    full_population_prevalence = (
        pl.scan_csv(fixture_path)
        .select(
            "survey_year",
            "Needed_Health_Care_Not_Received_Yes",
            "Hospital_Emergency_Room_Visits_None",
            "Hospital_Emergency_Room_Visits_1_time",
        )
        .group_by("survey_year")
        .agg(
            pl.len().alias("children"),
            (pl.col("Needed_Health_Care_Not_Received_Yes") == 1).sum().alias("Foregone care"),
            _any_emergency_visit.sum().alias("ED use, one or more visits"),
            _repeat_emergency_visits.sum().alias("ED use, two or more visits"),
        )
        .sort("survey_year")
        .collect()
    )
    return (full_population_prevalence,)


@app.cell
def _(
    autism_subset_prevalence,
    full_population_prevalence,
    outcome_names,
    period_labels,
    plt,
):
    outcome_colors = {
        "Foregone care": "#B85042",
        "ED use, one or more visits": "#003466",
        "ED use, two or more visits": "#5B8AA6",
        "Behaviour therapy": "#6E8B3D",
    }

    def draw_prevalence_by_period():
        """How common each outcome is, before any model is involved."""
        figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)

        autism_axis = axes[0]
        periods = autism_subset_prevalence["period"].to_list()
        for outcome in outcome_names:
            shares = [
                row[outcome] / row["children"]
                for row in autism_subset_prevalence.iter_rows(named=True)
            ]
            autism_axis.plot(
                periods,
                shares,
                marker="o",
                color=outcome_colors[outcome],
                linewidth=1.6,
                markersize=6,
                label=outcome,
            )
        autism_axis.set_xticks(periods)
        autism_axis.set_xticklabels(
            [period_labels.get(period, str(period)) for period in periods], fontsize=9
        )
        autism_axis.set_title("Children with autism, 2016 to 2023", fontsize=10)
        autism_axis.set_ylabel("share of children")
        autism_axis.set_ylim(0, 0.75)
        autism_axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        autism_axis.legend(frameon=False, fontsize=8, loc="center left")

        full_axis = axes[1]
        years = full_population_prevalence["survey_year"].to_list()
        # Behaviour therapy is absent here: the question is only asked of
        # children who already have a diagnosis.
        for outcome in outcome_names[:3]:
            shares = [
                row[outcome] / row["children"]
                for row in full_population_prevalence.iter_rows(named=True)
            ]
            full_axis.plot(
                years,
                shares,
                marker="o",
                color=outcome_colors[outcome],
                linewidth=1.6,
                markersize=6,
                label=outcome,
            )
        full_axis.set_xticks(years)
        full_axis.set_xticklabels([str(year) for year in years], fontsize=9)
        full_axis.set_title("All children, 2019 to 2020", fontsize=10)
        full_axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        full_axis.legend(frameon=False, fontsize=8, loc="upper left")

        figure.suptitle(
            "Children with autism go without needed care several times as often",
            fontsize=12,
        )
        figure.tight_layout()
        return figure

    draw_prevalence_by_period()
    return


@app.cell
def _(
    autism_subset_prevalence,
    full_population_prevalence,
    markdown_table,
    mo,
    period_labels,
):
    _autism_rows = [
        f"| {period_labels.get(row['period'], row['period'])} | {row['children']:,} "
        f"| {row['Foregone care'] / row['children']:.1%} "
        f"| {row['ED use, one or more visits'] / row['children']:.1%} "
        f"| {row['ED use, two or more visits'] / row['children']:.1%} "
        f"| {row['Behaviour therapy'] / row['children']:.1%} |"
        for row in autism_subset_prevalence.iter_rows(named=True)
    ]
    _autism_table = markdown_table(
        ["Period", "Children", "Foregone care", "ED use, 1+", "ED use, 2+", "Behaviour therapy"],
        ["left", "right", "right", "right", "right", "right"],
        _autism_rows,
    )
    _full_rows = [
        f"| {row['survey_year']} | {row['children']:,} "
        f"| {row['Foregone care'] / row['children']:.1%} "
        f"| {row['ED use, one or more visits'] / row['children']:.1%} "
        f"| {row['ED use, two or more visits'] / row['children']:.1%} |"
        for row in full_population_prevalence.iter_rows(named=True)
    ]
    _full_table = markdown_table(
        ["Survey year", "Children", "Foregone care", "ED use, 1+", "ED use, 2+"],
        ["left", "right", "right", "right", "right"],
        _full_rows,
    )
    _autism_total = int(autism_subset_prevalence["children"].sum())
    _full_total = int(full_population_prevalence["children"].sum())
    _autism_foregone = int(autism_subset_prevalence["Foregone care"].sum()) / _autism_total
    _full_foregone = int(full_population_prevalence["Foregone care"].sum()) / _full_total
    _foregone_ratio = _autism_foregone / _full_foregone
    mo.md(
        f"""
        ## 10. What the outcomes look like before any modelling

        Before running any model I want to know how common these things are,
        because that governs what a model can possibly find. An outcome that
        almost never happens leaves very little for the method to learn from,
        and section 2 already showed what a low base rate does to the measures.

        **{_autism_total:,} children with autism**, grouped into four two-year
        periods:

        {_autism_table}

        The same outcomes across **{_full_total:,} children of all kinds**, for
        the two survey years where I have that data assembled:

        {_full_table}

        Four things stand out.

        **Children with autism go without needed care far more often.**
        {_autism_foregone:.1%} of them, against {_full_foregone:.1%} of
        children generally, a ratio of about {_foregone_ratio:.0f} to 1. This
        is the single largest gap in the document and it is the kind of finding
        the manuscript exists to report. One caution on reading it: the two
        groups are measured over different spans, 2016 to 2023 against 2019 to
        2020, so this is a rough comparison rather than a matched one. The gap
        is far too large to be an artefact of that, but the precise ratio will
        move once the all-children data covers the same years.

        **Unmet need among children with autism has risen.** Foregone care runs
        around nine percent in the earliest period and around thirteen in the
        most recent.

        **The 2020 to 2021 period behaves differently from its neighbours.**
        Emergency room use dips there on both definitions while foregone care
        reaches its highest point, which points toward care being deferred
        rather than sought. The all-children table shows the same shape between
        2019 and 2020, from an entirely separate dataset, which makes it less
        likely to be a quirk of one sample. I would be careful about naming a
        cause from four two-year buckets and this notebook does not claim one.
        But a period that genuinely behaves unlike the others is precisely what
        the SOAK method exists to detect, so it is worth watching when the
        model results arrive.

        **The two emergency-room definitions are not interchangeable.** One or
        more visits describes roughly a fifth of these children. Two or more
        describes about one in fifteen. A model predicting the first is
        answering a much broader question than a model predicting the second,
        and I would not expect them to point at the same predictors.

        Every percentage above is unweighted, so these describe the children
        who were surveyed rather than the national population, for the reason
        given in section 8. The weighted versions will shift these numbers and
        may shift the gap in the first point, which is another reason that work
        matters.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 11. What runs now, and what is still being built

        Being precise here matters, because these analyses cover different
        populations over different spans and it would be easy to blur them
        together in a way that would not survive review.

        | Analysis | Population | Survey years | Status |
        | --- | --- | --- | --- |
        | Foregone care, ED use | all children | 2019 to 2020 | ready to run on the validated data |
        | All three outcomes | children with autism | 2016 to 2023 | ready to run |
        | Foregone care, ED use | all children | 2016 to 2023 | needs a dataset that does not exist yet |
        | Everything | either | 2024 | blocked on one missing file |

        The third row is the real gap, and I want to flag it rather than let it
        surface later. Running the all-children analysis across every survey
        year needs a combined dataset that exists on no machine we have. Vince
        built the two-year version, and the code that would build the full one
        has to be ported before it can be rebuilt. That is careful work, and
        rushing it would quietly corrupt everything downstream rather than
        failing loudly. It is on the critical path for the manuscript.

        The 2024 data is waiting on a single file from a collaborator, which is
        queued and should be quick once it arrives.

        **What comes next.** Model results for the first two rows, then the
        survey-weighted analysis that turns statements about this sample into
        statements about children in the United States. And, if section 6
        persuaded you as it persuaded me, an equal-sized-training-set run
        before we say anything about how the survey years differ from one
        another.
        """
    )
    return


if __name__ == "__main__":
    app.run()
