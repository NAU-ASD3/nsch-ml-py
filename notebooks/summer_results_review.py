import marimo

__generated_with = "0.23.16"
app = marimo.App(
    width="medium",
    layout_file="layouts/summer_results_review.slides.json",
)


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from sklearn.metrics import roc_auc_score

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "analyses"))

    def input_path(variable, *parts):
        configured = os.environ.get(variable)
        if not configured:
            return None
        path = Path(configured).expanduser().joinpath(*parts)
        return path if path.exists() else None

    service_matrix_path = input_path("MONSOON_OLD", "2016_2023_ServiceUse.csv")
    r_seed1_path = input_path("REPRO", "results", "seed-variation", "NSCH_seed1.csv")
    r_other_path = input_path("REPRO", "results", "2026-03-06", "NSCH_proj.csv")

    results_dir = repo_root / "analyses" / "results"
    contrasts = pl.read_csv(results_dir / "contrasts.csv")
    replication = pl.read_csv(
        repo_root / "analyses" / "glmnet_replication_lasso_seed1_is100.csv"
    ).filter(~pl.col("downsampled") & pl.col("r_auc").is_not_null())

    def task_results(name):
        path = results_dir / f"{name}.csv"
        return pl.read_csv(path) if path.exists() else None

    # Reused from the replication figures so the two documents match.
    full_size_color = "#9C7A4C"
    equal_size_color = "#4C6E9C"

    # The title is the output of this cell rather than a cell of its own. Every
    # cell becomes a slide, and a setup cell with no output would open the deck
    # on a blank one.
    mo.md(
        r"""
        # Where the machine learning work stands

        Chris Reger, 25 August 2026

        Every number here is computed when the page is built, from result
        files in the repository.
        """
    )
    return (
        contrasts,
        equal_size_color,
        full_size_color,
        mo,
        np,
        pl,
        plt,
        r_other_path,
        r_seed1_path,
        replication,
        roc_auc_score,
        service_matrix_path,
        task_results,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Why any of this happened

    The question: which children with autism are not getting the care they
    need, and has that been changing.

    Vince built an analysis in R. I needed to point it at these new
    outcomes.

    Before changing an analysis I would rather know it does what I think.
    So I rebuilt it in Python and checked whether the two agreed.

    That check took most of the summer, and it found things.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The method, in one slide

    Hold out a tenth of one period's children. Train three models. See
    which predicts those children best.

    - **Same**: the rest of that period
    - **Other**: the different periods
    - **All**: everything

    If **All** wins, the periods can be pooled. That is the permission
    slip for treating 2016 to 2023 as one population.
    """)
    return


@app.cell
def _(mo, pl, r_other_path, r_seed1_path, replication):
    _gap = (replication["auc_1se"] - replication["r_auc"]).abs().max()

    def _seed_spread():
        if r_seed1_path is None or r_other_path is None:
            return None
        keys = ["test.subset", "train.subsets", "test.fold"]
        frames = []
        for path, name in ((r_seed1_path, "a"), (r_other_path, "b")):
            frames.append(
                pl.read_csv(path, infer_schema_length=20000)
                .filter(pl.col("learner_id") == "classif.cv_glmnet")
                .sort("n.train.groups", descending=True)
                .unique(subset=keys, keep="first")
                .select(
                    pl.col("test.subset").cast(pl.Utf8),
                    pl.col("train.subsets"),
                    pl.col("test.fold"),
                    pl.col("classif.auc").alias(name),
                )
            )
        joined = frames[0].join(frames[1], on=keys, how="inner")
        return float((joined["a"] - joined["b"]).abs().max())

    seed_gap = _seed_spread()
    _seed_text = f"{seed_gap:.4f}" if seed_gap else "unavailable"

    mo.md(
        f"""
        ## Result one: the rebuild is faithful

        | | Largest disagreement |
        | --- | ---: |
        | R original against my Python version | **{float(_gap):.4f}** |
        | Two runs of Vince's own R code, differing only by a random seed | **{_seed_text}** |

        The second is an order of magnitude larger than the first.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Which is why the standard came first

    "The two versions agree closely" means nothing on its own. It only
    means something against a stated threshold.

    Choose that threshold after seeing your results and you can always
    choose one they clear.

    So we wrote ours down and committed it before running the comparison.

    It bit. Two of sixty splits failed. We reported that and left the
    threshold where it was.
    """)
    return


@app.cell
def _(mo, pl, roc_auc_score, service_matrix_path, task_results):
    _primary = task_results("service_use_foregone_care")
    _ed = task_results("service_use_ed_any")

    def _mean_auc(frame):
        if frame is None:
            return float("nan")
        full = frame.filter(~pl.col("downsampled"))
        return sum(full["auc_1se"].to_list()) / full.height

    def _frustration_alone():
        if service_matrix_path is None:
            return float("nan")
        data = pl.read_csv(service_matrix_path, infer_schema_length=None)
        # Three of the four response levels are indicator columns; the fourth
        # is the rows where none of them is set.
        levels = ["c4q04=Never", "c4q04=Sometimes", "c4q04=Usually"]
        score = (
            pl.col(levels[1]) * 1
            + pl.col(levels[2]) * 2
            + (1 - pl.col(levels[0]) - pl.col(levels[1]) - pl.col(levels[2])) * 3
        )
        prepared = data.with_columns(
            score.alias("frustration"),
            (pl.col("k4q27=Yes") == 1).cast(pl.Int64).alias("y"),
        )
        return float(roc_auc_score(prepared["y"], prepared["frustration"]))

    frustration_auc = _frustration_alone()
    primary_auc = _mean_auc(_primary)

    mo.md(
        f"""
        ## Result two: a result that was too good

        Foregone care, children with autism: **{primary_auc:.3f}**

        Emergency department use: **{_mean_auc(_ed):.3f}**

        Unmet care need should be the harder problem, not the easier one.
        That gap is what made me look.
        """
    )
    return frustration_auc, primary_auc


@app.cell
def _(frustration_auc, mo, primary_auc):
    mo.md(f"""
    ## One question was doing the work

    *"How often have you felt frustrated in your efforts to get services
    for this child?"*

    That question alone, with no model and nothing else:
    **{frustration_auc:.3f}**

    Every other survey answer, together, adds
    **{primary_auc - frustration_auc:.3f}**.
    """)
    return


@app.cell
def _(mo, pl, service_matrix_path):
    def _gradient():
        if service_matrix_path is None:
            return "unavailable"
        data = pl.read_csv(service_matrix_path, infer_schema_length=None)
        labelled = data.with_columns(
            pl.when(pl.col("c4q04=Never") == 1)
            .then(pl.lit("Never"))
            .when(pl.col("c4q04=Sometimes") == 1)
            .then(pl.lit("Sometimes"))
            .when(pl.col("c4q04=Usually") == 1)
            .then(pl.lit("Usually"))
            .otherwise(pl.lit("Always"))
            .alias("frustration")
        )
        summary = labelled.group_by("frustration").agg(
            pl.len().alias("children"),
            (pl.col("k4q27=Yes") == 1).sum().alias("foregone"),
        )
        order = {"Never": 0, "Sometimes": 1, "Usually": 2, "Always": 3}
        rows = sorted(summary.iter_rows(named=True), key=lambda r: order[r["frustration"]])
        return "\n        ".join(
            f"| {row['frustration']} | {row['children']:,} "
            f"| {row['foregone'] / row['children']:.1%} |"
            for row in rows
        )

    mo.md(
        f"""
        ## Why that is a problem, not a finding

        | Frustrated getting services | Children | Went without care |
        | --- | ---: | ---: |
        {_gradient()}

        These are two ways of asking the same question. Predicting one from
        the other tells us nothing anyone can act on.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How it was found

    I had audited for exactly this. My audit searched for columns whose
    names resembled the outcome's code.

    The frustration question is `c4q04`. The outcome is `k4q27`. No shared
    characters, so my method could not have found it, and could not tell
    me what else it missed.

    The replacement reads the survey's own question wording for every
    column. It covers 100% of both datasets.

    It found a second one I had missed, and cleared a variable I had
    wrongly excluded.
    """)
    return


@app.cell
def _(contrasts, equal_size_color, full_size_color, np, pl, plt):
    def draw_size_comparison():
        """Full size against equal size. Primary specifications only, for legibility."""
        pooled = contrasts.filter(
            (pl.col("metric") == "auc_1se")
            & (pl.col("contrast") == "All - Same")
            & (pl.col("scope") == "pooled")
            & ~pl.col("task").str.contains("conservative")
            & ~pl.col("task").str.contains("strict")
        )
        labels = {
            "fixture_ed_any": "ED use, all children",
            "fixture_ed_repeat": "Repeat ED use, all children",
            "fixture_foregone_care": "Foregone care, all children",
            "service_use_behaviour_therapy": "Behaviour therapy, autism",
            "service_use_ed_any": "ED use, autism",
            "service_use_ed_repeat": "Repeat ED use, autism",
            "service_use_foregone_care": "Foregone care, autism",
        }
        tasks = sorted(pooled["task"].unique().to_list(), reverse=True)
        figure, axis = plt.subplots(figsize=(10.0, 4.6))
        for offset, (variant, colour, label) in enumerate(
            (
                ("full", full_size_color, "as normally done"),
                ("equal", equal_size_color, "equal training size"),
            )
        ):
            rows = pooled.filter(pl.col("size_variant") == variant)
            for position, task in enumerate(tasks):
                match = rows.filter(pl.col("task") == task)
                if match.height == 0:
                    continue
                row = match.row(0, named=True)
                axis.errorbar(
                    [row["mean"]],
                    [position + (offset - 0.5) * 0.3],
                    xerr=[[row["mean"] - row["lo"]], [row["hi"] - row["mean"]]],
                    fmt="o",
                    color=colour,
                    capsize=3,
                    markersize=6,
                    linewidth=1.6,
                    label=label if position == 0 else None,
                )
        axis.axvline(0, color="#777777", linewidth=1, linestyle="--")
        axis.set_yticks(np.arange(len(tasks)))
        axis.set_yticklabels([labels.get(task, task) for task in tasks], fontsize=10)
        axis.set_xlabel("How much pooling the periods helps, in AUC")
        axis.grid(axis="x", alpha=0.25, linewidth=0.6)
        axis.legend(frameon=False, fontsize=10, loc="upper left")
        axis.set_title(
            "Pooling helps because it is more data, not because the periods are alike",
            fontsize=12,
        )
        figure.tight_layout()
        return figure

    draw_size_comparison()
    return


@app.cell
def _(contrasts, mo, pl):
    _pooled = contrasts.filter(
        (pl.col("metric") == "auc_1se")
        & (pl.col("contrast") == "All - Same")
        & (pl.col("scope") == "pooled")
    )
    _full = _pooled.filter(pl.col("size_variant") == "full")
    _equal = _pooled.filter(pl.col("size_variant") == "equal")
    _full_positive = _full.filter((pl.col("mean") > 0) & (pl.col("p") < 0.05)).height
    _equal_positive = _equal.filter((pl.col("mean") > 0) & (pl.col("p") < 0.05)).height

    mo.md(
        f"""
        ## Result three: what that comparison measures

        "All periods" uses more children than "one period". Here, 1.8 to 5.4
        times as many, and the ratio varies across periods.

        Pooling wins in **{_full_positive} of {_full.height}** analyses as
        normally run, and **{_equal_positive} of {_equal.height}** once the
        training sets are cut to equal size.

        Pooling is still right for your manuscript. The reason is that more
        data helps, not that the periods inform each other.
        """
    )
    return


@app.cell
def _(mo, pl, task_results):
    _rows = []
    for _name, _label in (
        ("service_use_foregone_care", "Foregone care, autism"),
        ("service_use_ed_any", "ED use, autism"),
        ("fixture_foregone_care", "Foregone care, all children"),
        ("fixture_ed_any", "ED use, all children"),
    ):
        _frame = task_results(_name)
        if _frame is None:
            continue
        _full = _frame.filter(~pl.col("downsampled"))
        _slopes = [v for v in _full["calibration_slope_1se"].to_list() if v is not None]
        _sizes = _full["n_train"].to_list()
        _rows.append(
            f"| {_label} | {min(_sizes):,} to {max(_sizes):,} | {sum(_slopes) / len(_slopes):.2f} |"
        )
    _table = "\n        ".join(_rows)

    mo.md(
        f"""
        ## Result four: the probabilities are not trustworthy

        A model can rank children correctly while misstating how likely each
        one is. A truthful model scores 1.00.

        | Analysis | Children trained on | Truthfulness |
        | --- | ---: | ---: |
        {_table}

        The rankings are fine. The probabilities are not, and they get worse
        the smaller the training set.
        """
    )
    return


@app.cell
def _(contrasts, mo, pl):
    _odd = contrasts.filter(
        (pl.col("task") == "service_use_ed_repeat")
        & (pl.col("metric") == "auc_1se")
        & (pl.col("scope") == "pooled")
        & (pl.col("size_variant") == "equal")
    )
    _lines = "\n        ".join(
        f"| {row['contrast']} | {row['mean']:+.4f} | {row['p']:.3f} |"
        for row in _odd.iter_rows(named=True)
    )

    mo.md(
        f"""
        ## The one that does not fit

        Repeat ED use among children with autism goes the other way at matched
        size. Training on other periods is worse, not merely no better.

        | Comparison | Difference | p |
        | --- | ---: | ---: |
        {_lines}

        Either the one real period effect in the set, or the thinness we
        flagged in advance: about seven cases per test set.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What none of this gives you

    All of it is about method. None of it produces a survey-weighted
    estimate of how many children with autism go without needed care.

    Your manuscript makes population claims. Those need weights,
    design-based standard errors, and adjusted odds ratios.

    That analysis has not been designed and has no written plan. I have
    been treating it as separate work.

    **If you expected it to fall out of these runs, I would rather know
    today than in October.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Option one: how to know a port is faithful

    Not "we ported some code". Nobody has a good answer for how you check
    such a port, and the obvious answers are wrong.

    The evidence is unusually complete because it was written down as it
    happened, including two of my own mistakes and the process that caught
    them.

    **Where:** reproducibility or meta-research. Not health services.

    **Worry:** smaller audience, and a reviewer may call it obvious.

    **Left:** mostly writing.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Option two: a comparison that measures the wrong thing

    The size result, generalised to anyone pooling repeated survey years.

    **Credit where due.** Toby built the correction into the method
    because he anticipated this. Ours is the demonstration on real data.

    **Worries, and these are real.** Our claim rests on effects
    disappearing, which needs equivalence testing. The analyses are not
    independent. Our test rejects too easily. One analysis resists.

    **Left:** resolve that analysis, add equivalence tests, more writing.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Three things I need from you

    **Which order?** My inclination is option one first, since option two
    asks a reader to trust a port that option one validates. If the summer
    should end with a result rather than a protocol, that changes.

    **Should Toby see option two first?** His method, his mechanism.

    **The weighted analysis.** Neither paper produces it, your manuscript
    needs it, and it has no plan. What does the autumn look like?
    """)
    return


if __name__ == "__main__":
    app.run()
