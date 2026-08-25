"""Label every feature column from the survey's own .do file, and flag the risky ones.

The leak audit that missed `c4q04` matched column names against the outcome's
variable code. `c4q04` shares no characters with `k4q27`, so that method could
not have found it, and it cannot say what else it missed among the other 291
features. Fixing the one variable would leave the method just as broken and we
would find the next leak the same way, by being surprised at a result.

This reads the survey's `.do` file, attaches the official label to every
feature column, and prints them grouped so the audit is done against what the
questions actually ask rather than against what their codes look like.

Two matrices, two naming schemes, one join:

  the autism-subset matrix names columns `<stem>=<level>`, so the stem before
  the `=` is the survey variable and joins to `label var` directly

  the full-population matrix spells the label out, so
  `Needed_Health_Care_Not_Received_Yes` is `label var k4q27 "Needed Health Care
  Not Received"` with punctuation replaced and the level appended

Matching ignores punctuation entirely on both sides. An earlier version
collapsed runs of separators, which failed on every column built from a label
containing " - ", since the column keeps one underscore per character and the
label yields only one. That single detail cost 45% of the coverage on the
full-population matrix, so the join now compares letters and digits alone.

Columns that match nothing are reported rather than dropped. A feature whose
question we cannot name is a feature we cannot audit, and that is worth seeing.

The flags are a reading order, not a verdict. The output worth acting on is the
full labelled list, and the rule the analysis plan adopts is that every
exclusion cites a label from it.

Run from the repository root::

    uv run python analyses/audit_feature_constructs.py \\
        --matrix service_use \\
        --data "$MONSOON_OLD/2016_2023_ServiceUse.csv" \\
        --do-file "$NSCH/inst/extdata/nsch_2024_topical.do"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl
from outcomes import matrix_or_exit

LABEL_VAR = re.compile(r'^\s*label\s+var\s+(\S+)\s+"([^"]*)"', re.MULTILINE)

# Harmonization renames a few source columns, so the matrix knows a variable
# under a name the .do file does not. These mirror `rename_columns` in the nsch
# package's variable-config.json; without them the renamed columns look
# unlabelled and drop out of the audit entirely.
MATRIX_TO_SURVEY_STEM = {
    "k4q02_r": "gowhensick",
    "family": "family_r",
    "diabetes": "k2q41a",
}

# Questions about the process of obtaining care in the same twelve-month window
# as an access outcome. These measure the same care-seeking episode the outcome
# measures, which is where a construct leak sits.
CARE_SEEKING_TERMS = (
    "not received",
    "frustrated",
    "need a referral",
    "arrange or coordinate",
    "cover services",
    "cover mental",
    "allow to see provider",
    "unmet",
    "denied",
    # "delayed", not "delay": the latter matches "Developmental Delay", which is
    # a condition the child has rather than a barrier to care.
    "delayed",
    "difficult to get",
    "problem getting",
)

# Broader stems, kept deliberately blunt so the audit keeps some recall against
# constructs nobody thought to name. Printed separately because precision here
# is poor: "Difficulty Toothaches Past 12 Months" is a symptom, not a barrier.
REVIEW_TERMS = (
    "problem",
    "difficult",
    "cost",
    "afford",
    "coordinat",
    "referral",
    "cover",
    "insurance",
    "needed",
)

RULE = "=" * 78


def parse_do_labels(do_paths: list[Path]) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Union every `label var` across the given .do files, earliest file winning.

    A harmonized matrix cannot be audited against one survey year. Its column
    names come from whichever year the harmonization settled on, so `k4q02_r`
    is the 2019 name while `eyedoctor` is the 2024 one, and no single file
    knows both. Reading several and taking the union covers the matrix.

    The second return value collects every distinct wording seen for a stem.
    A question whose label changed between years may be a question whose
    meaning changed, which is worth surfacing rather than silently resolving.
    """
    labels: dict[str, str] = {}
    variants: dict[str, set[str]] = {}
    for do_path in do_paths:
        text = do_path.read_text(encoding="utf-8", errors="replace")
        for match in LABEL_VAR.finditer(text):
            stem, description = match.group(1), match.group(2)
            labels.setdefault(stem, description)
            variants.setdefault(stem, set()).add(description)
    return labels, variants


def letters_and_digits(text: str) -> str:
    """Punctuation-free, case-free form, for comparing a label to a column name."""
    return re.sub(r"[^0-9A-Za-z]", "", text).lower()


def tail_after(column: str, consumed: int) -> str:
    """The part of a column name left after the label's characters are matched."""
    seen = 0
    for position, character in enumerate(column):
        if character.isalnum():
            seen += 1
            if seen == consumed:
                return column[position + 1 :].strip("_ -")
    return ""


def resolve_by_prefix(column: str, normalised: list[tuple[str, str]]) -> tuple[str, str]:
    """Longest label whose letters prefix this column's letters."""
    target = letters_and_digits(column)
    for candidate, stem in normalised:
        if target.startswith(candidate):
            return stem, tail_after(column, len(candidate))
    return "", ""


def resolve_by_stem(column: str, labels: dict[str, str]) -> tuple[str, str]:
    """Stem before the `=`, falling back to a rename alias only if needed.

    The literal name is tried first. Applying the alias unconditionally breaks
    the years where the matrix already carries the survey's own name: the
    column `k4q02_r` is `gowhensick` only from 2023 onward.
    """
    head, _, level = column.partition("=")
    if head in labels:
        return head, level
    alias = MATRIX_TO_SURVEY_STEM.get(head)
    if alias and alias in labels:
        return alias, level
    return "", ""


def classify(label: str) -> str:
    """Which reading tier a label falls into."""
    lowered = label.lower()
    if any(term in lowered for term in CARE_SEEKING_TERMS):
        return "care-seeking"
    if any(term in lowered for term in REVIEW_TERMS):
        return "review"
    return "circumstance"


def print_group(frame: pl.DataFrame, heading: str, note: str) -> None:
    print(RULE)
    print(heading)
    print(note)
    current = ""
    for row in frame.iter_rows(named=True):
        if row["stem"] != current:
            current = row["stem"]
            print(f'\n  {row["stem"]:<16} "{row["label"]}"')
        print(f"      {row['level'] or '(continuous)'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, help="fixture or service_use")
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--do-file",
        required=True,
        action="append",
        help="a survey year's .do file; repeat to cover a multi-year matrix",
    )
    parser.add_argument("--outcome", default=None, help="restrict to one outcome's features")
    parser.add_argument("--out", default=None, help="where to write the labelled list")
    args = parser.parse_args()

    data_path = Path(args.data)
    do_paths = [Path(name) for name in args.do_file]
    missing = [str(path) for path in [data_path, *do_paths] if not path.is_file()]
    if missing:
        for path in missing:
            print(f"REFUSED: no file at {path}")
        return 1

    spec = matrix_or_exit(args.matrix)
    labels, label_variants = parse_do_labels(do_paths)
    columns = pl.read_csv(data_path, n_rows=1).columns

    if args.outcome:
        outcome = spec.outcome_or_exit(args.outcome)
        features = spec.feature_columns(columns, outcome)
    else:
        features = [name for name in columns if name not in spec.non_feature_columns]

    by_prefix = spec.key == "fixture"
    normalised = sorted(
        ((letters_and_digits(text), stem) for stem, text in labels.items()),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    rows = []
    for column in features:
        stem, level = (
            resolve_by_prefix(column, normalised) if by_prefix else resolve_by_stem(column, labels)
        )
        label_text = labels.get(stem, "") if stem else ""
        rows.append(
            {
                "feature": column,
                "stem": stem,
                "level": level,
                "label": label_text,
                "tier": classify(label_text) if stem else "unlabelled",
            }
        )
    frame = pl.DataFrame(rows)

    print(RULE)
    print(f"Matrix    {data_path}")
    print(f"          {spec.label}")
    print(f"Labels    {len(labels)} from {len(do_paths)} file(s)")
    print(f"Features  {len(features)}")
    counts = dict(frame.group_by("tier").agg(pl.len().alias("n")).iter_rows())
    labelled = len(features) - counts.get("unlabelled", 0)
    print(f"Labelled  {labelled} of {len(features)} ({labelled / len(features):.0%})")

    print_group(
        frame.filter(pl.col("tier") == "care-seeking").sort(["stem", "level"]),
        "Tier 1: questions about obtaining care in the same window as the outcome.",
        "These measure the same care-seeking episode. Excluding these is the rule.",
    )
    print_group(
        frame.filter(pl.col("tier") == "review").sort(["stem", "level"]),
        "Tier 2: adjacent wording, poor precision, read before deciding.",
        "Includes symptoms phrased as 'Difficulty ...', which are not barriers.",
    )

    unlabelled = frame.filter(pl.col("tier") == "unlabelled")
    if unlabelled.height:
        print(RULE)
        print(f"No label found for {unlabelled.height} features. These cannot be audited")
        print("from these .do files and must be identified another way before being trusted.")
        for row in unlabelled.iter_rows(named=True):
            print(f"  {row['feature']}")

    drifted = sorted(
        stem
        for stem in frame.filter(pl.col("stem") != "")["stem"].unique().to_list()
        if len(label_variants.get(stem, set())) > 1
    )
    if drifted:
        print(RULE)
        print("These features are labelled differently in different survey years. A")
        print("question whose wording changed may be a question whose meaning changed.")
        for stem in drifted:
            print(f"\n  {stem}")
            for wording in sorted(label_variants[stem]):
                print(f'      "{wording}"')

    out_path = (
        Path(args.out) if args.out else Path("analyses/feature-audit") / f"{spec.key}_features.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort(["tier", "stem", "level"]).write_csv(out_path)
    print(RULE)
    print(f"wrote {out_path}")
    print("Every exclusion in the analysis plan should cite a label from this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
