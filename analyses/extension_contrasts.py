"""SOAK contrasts for the extension outcomes, following the pre-registered plan.

Every contrast reported so far has been computed by hand at a terminal. This
puts them in the repository, so the numbers in `docs/` and in the notebooks can
be traced to code rather than to a command someone typed once.

The statistics come from `soak_ttests.py`; only the schema handling is new.
The replication's results carry an R column beside every Python column, and
`soak_ttests` is built around comparing the two. These results have no R
counterpart, four subsets rather than two, and an `is_equal_size` column.

**The script enforces the plan's reporting rules rather than leaving them to
whoever writes the table.**

Confirmatory, one per task: All minus Same, at full training size, pooled
across subsets into a single test. Full size is right here because the larger
training set is the treatment, not a confound: the question is whether more
data from other periods helps.

Exploratory: per-subset contrasts, and Other minus Same.

Other minus Same is reported **only at equal training size**, and the
full-size version is not printed at all. On the autism-subset matrix `Other`
trains on between 1.8 and 5.4 times as many children as `Same`, and the ratio
varies across periods, so at full size that contrast measures training-set
volume rather than period transferability. Printing both would invite a reader
to average two numbers that do not mean the same thing. See the 12 August
amendment in docs/extension-analysis-plan.md.

Run from the repository root::

    uv run python analyses/extension_contrasts.py
    uv run python analyses/extension_contrasts.py --results analyses/results/service_use_ed_any.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from soak_ttests import ALPHA, paired_test, significance_call

DEFAULT_RESULTS_DIR = Path("analyses/results")
RULE = "=" * 78


@dataclass(frozen=True)
class ContrastMetric:
    """One quantity to contrast, and which direction counts as better."""

    column: str
    label: str
    higher_is_better: bool


METRICS = (
    ContrastMetric("auc_1se", "AUC", higher_is_better=True),
    ContrastMetric("percent_error_1se", "percent error", higher_is_better=False),
    ContrastMetric("brier_1se", "Brier score", higher_is_better=False),
)

# (name, first source, second source, which size variant it may be read from)
CONTRAST_SPECS = (
    ("All - Same", "all", "same", "full"),
    ("All - Same", "all", "same", "equal"),
    ("Other - Same", "other", "same", "equal"),
)


def completed_splits(results_path: Path) -> tuple[int | None, str]:
    """How many splits the provenance record says this run wrote.

    A results file is checkpointed every 25 splits, so a run that was killed
    leaves a file that reads as valid and is simply short. Nothing in the file
    itself says how long it should be. The provenance record is written once,
    at the end, and names the split count, so its absence means the run never
    finished and its count is what the file should contain.

    This matters more than it sounds. A half-finished fixture task holds only
    its 2019 splits, so every contrast computed from it silently becomes a
    single-year result reported as though it covered both.
    """
    provenance_path = results_path.with_suffix(".provenance.csv")
    if not provenance_path.is_file():
        return None, (
            "no provenance record, so this run did not finish. Rerun it, or delete "
            "the partial results file."
        )
    record = pl.read_csv(provenance_path)
    fields = dict(zip(record["field"].to_list(), record["value"].to_list(), strict=True))
    declared = fields.get("n_splits")
    if declared is None:
        return None, "provenance record has no n_splits field"
    return int(declared), ""


def result_files(results_dir: Path, explicit: list[str] | None) -> list[Path]:
    if explicit:
        return [Path(name) for name in explicit]
    return sorted(
        path
        for path in results_dir.glob("*.csv")
        if not path.name.endswith(".provenance.csv") and path.name != "contrasts.csv"
    )


def size_filter(frame: pl.DataFrame, variant: str) -> pl.DataFrame:
    """Rows belonging to one size variant.

    `downsampled` alone does not identify the equal-size arm: a source already
    at the target size is not duplicated by the splitter, so where `Same` is
    the smallest source its equal-size arm is its full split. The runner
    records that in `is_equal_size`, which is why this reads that column
    rather than re-deriving the rule.
    """
    if variant == "full":
        return frame.filter(~pl.col("downsampled"))
    return frame.filter(pl.col("is_equal_size").fill_null(value=False))


def paired_columns(frame: pl.DataFrame, metric: ContrastMetric) -> pl.DataFrame:
    """One row per (subset, fold), with a column per training source."""
    return (
        frame.select("test_subset", "fold", "train_source", metric.column)
        .pivot(on="train_source", index=["test_subset", "fold"], values=metric.column)
        .sort(["test_subset", "fold"])
    )


def contrast_rows(
    task: str,
    frame: pl.DataFrame,
    metric: ContrastMetric,
) -> list[dict[str, Any]]:
    """Every contrast the plan permits for one task and one metric."""
    rows: list[dict[str, Any]] = []
    for name, first, second, variant in CONTRAST_SPECS:
        wide = paired_columns(size_filter(frame, variant), metric)
        if first not in wide.columns or second not in wide.columns:
            continue
        usable = wide.drop_nulls([first, second])
        dropped = wide.height - usable.height
        if usable.height < 2:
            continue

        first_values = usable[first].to_numpy().astype(float)
        second_values = usable[second].to_numpy().astype(float)

        for subset in [*sorted(usable["test_subset"].unique().to_list()), None]:
            if subset is None:
                a, b = first_values, second_values
                scope, label = "pooled", "pooled"
            else:
                mask = (usable["test_subset"] == subset).to_numpy()
                a, b = first_values[mask], second_values[mask]
                scope, label = "per-subset", str(subset)
            if len(a) < 2:
                continue
            result = paired_test(a, b)
            rows.append(
                {
                    "task": task,
                    "metric": metric.column,
                    "metric_label": metric.label,
                    "contrast": name,
                    "size_variant": variant,
                    "scope": scope,
                    "test_subset": label,
                    # The confirmatory test named in the plan: All minus Same,
                    # full size, pooled. Everything else is exploratory and is
                    # labelled so wherever it appears.
                    "role": (
                        "confirmatory"
                        if name == "All - Same"
                        and variant == "full"
                        and scope == "pooled"
                        and metric.column == "auc_1se"
                        else "exploratory"
                    ),
                    "n_pairs": len(a),
                    "dropped_pairs": dropped if subset is None else 0,
                    "mean": result["mean"],
                    "lo": result["lo"],
                    "hi": result["hi"],
                    "p": result["p"],
                    "verdict": significance_call(
                        result["mean"], result["p"], metric.higher_is_better
                    ),
                }
            )
    return rows


def column_mean(frame: pl.DataFrame, column: str) -> float:
    """Mean of a column as a plain float, or nan.

    Series.mean() is typed as a union spanning every dtype polars can hold, so
    it does not narrow to a number. Going through to_list() keeps the
    arithmetic honest without a cast that would suppress a real error.
    """
    if column not in frame.columns:
        return float("nan")
    values = [float(value) for value in frame[column].to_list() if value is not None]
    return sum(values) / len(values) if values else float("nan")


def describe_task(frame: pl.DataFrame) -> str:
    """A one-line summary of what a task's model achieved, before contrasts."""
    full = frame.filter(~pl.col("downsampled"))
    parts = [f"auc {column_mean(full, 'auc_1se'):.4f}"]
    for column, name, spec in (
        ("brier_1se", "brier", ".4f"),
        ("calibration_slope_1se", "slope", ".2f"),
        ("n_nonzero_1se", "kept", ".0f"),
    ):
        if column in full.columns:
            parts.append(f"{name} {column_mean(full, column):{spec}}")
    return "   ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--results", nargs="*", default=None, help="specific result files")
    parser.add_argument("--out", default=None, help="tidy contrast CSV")
    args = parser.parse_args()

    paths = result_files(Path(args.results_dir), args.results)
    if not paths:
        print(f"REFUSED: no result files under {args.results_dir}")
        return 1

    all_rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for path in paths:
        frame = pl.read_csv(path)
        required = {"test_subset", "fold", "train_source", "downsampled", "auc_1se"}
        absent = required - set(frame.columns)
        if absent:
            print(f"skipping {path.name}: missing {', '.join(sorted(absent))}")
            continue
        task = path.stem
        declared, problem = completed_splits(path)
        if declared is None:
            print(RULE)
            print(f"{task}")
            print(f"  SKIPPED: {problem}")
            skipped.append(task)
            continue
        if frame.height != declared:
            print(RULE)
            print(f"{task}")
            print(
                f"  SKIPPED: the file holds {frame.height} splits but its provenance "
                f"declares {declared}. This is a checkpoint from an interrupted run."
            )
            skipped.append(task)
            continue
        print(RULE)
        print(f"{task}")
        print(f"  {describe_task(frame)}")
        rows = contrast_rows(task, frame, METRICS[0])
        for metric in METRICS[1:]:
            rows += contrast_rows(task, frame, metric)
        all_rows += rows

        for role in ("confirmatory", "exploratory"):
            selected = [row for row in rows if row["role"] == role and row["metric"] == "auc_1se"]
            if not selected:
                continue
            print(f"  {role}, AUC")
            for row in selected:
                star = " *" if row["p"] < ALPHA else "  "
                print(
                    f"    {row['contrast']:<12} {row['size_variant']:<5} "
                    f"{row['test_subset']:<8} {row['mean']:+.4f} "
                    f"[{row['lo']:+.4f}, {row['hi']:+.4f}]  p {row['p']:.4f}{star}"
                    f"  n {row['n_pairs']}"
                )

    if not all_rows:
        print("REFUSED: no contrasts computed")
        return 1

    out_path = Path(args.out) if args.out else Path(args.results_dir) / "contrasts.csv"
    pl.DataFrame(all_rows).write_csv(out_path)
    print(RULE)
    if skipped:
        print(f"skipped {len(skipped)} incomplete task(s): {', '.join(skipped)}")
    print(
        f"wrote {out_path}   ({len(all_rows)} contrasts across {len(paths) - len(skipped)} tasks)"
    )
    print("Confirmatory rows are All minus Same on AUC, full size, pooled across")
    print("subsets. Everything else is exploratory and labelled so in the file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
