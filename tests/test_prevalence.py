"""Tests for nsch_ml.prevalence, on small synthetic frames only."""

import math
from pathlib import Path

import polars as pl
import pytest

from nsch_ml.prevalence import (
    GROUP_CURRENT,
    GROUP_CURRENT_UNKNOWN,
    GROUP_EVER_UNKNOWN,
    GROUP_NEVER,
    GROUP_NOT_CURRENT,
    add_indicators,
    classify_autism,
    find_topical_file,
    read_do_labels,
    weighted_proportion,
    weighted_total,
)


def _children(ever: list[str], current: list[str], ages: list[int]) -> pl.DataFrame:
    n = len(ever)
    return pl.DataFrame(
        {
            "year": [2020] * n,
            "hhid": list(range(1, n + 1)),
            "fipsst": [4] * n,
            "stratum": ["1"] * n,
            "fwc": [1.0] * n,
            "sc_age_years": ages,
            "k2q35a": ever,
            "k2q35b": current,
        }
    )


def test_each_answer_pattern_lands_in_its_group() -> None:
    children = _children(["1", "1", "1", "2", "m", "m"], ["1", "2", "m", "l", "m", "2"], [5] * 6)
    result = classify_autism(children)
    assert result["autism_group"].to_list() == [
        GROUP_CURRENT,
        GROUP_NOT_CURRENT,
        GROUP_CURRENT_UNKNOWN,
        GROUP_NEVER,
        GROUP_EVER_UNKNOWN,
        GROUP_EVER_UNKNOWN,
    ]


def test_unanticipated_answer_pattern_is_refused() -> None:
    # "No" on the first item with a real answer on the follow-up should never
    # occur; if a release ever contains it, the analysis must stop.
    children = _children(["2"], ["1"], [5])
    with pytest.raises(ValueError, match="Unanticipated"):
        classify_autism(children)


def test_children_outside_age_band_get_zero_indicators() -> None:
    children = classify_autism(_children(["1", "1", "1"], ["1", "1", "1"], [2, 3, 17]))
    result = add_indicators(children, 3, 17)
    assert result["in_age"].to_list() == [0, 1, 1]
    assert result["ever_yes"].to_list() == [0, 1, 1]
    assert result["current_yes"].to_list() == [0, 1, 1]


def test_missing_follow_up_counts_for_ever_but_not_for_current() -> None:
    children = classify_autism(_children(["1", "2"], ["m", "l"], [8, 8]))
    result = add_indicators(children, 3, 17)
    assert result["ever_valid"].to_list() == [1, 1]
    assert result["ever_yes"].to_list() == [1, 0]
    assert result["current_valid"].to_list() == [0, 1]
    assert result["current_yes"].to_list() == [0, 0]


def test_weighted_proportion_matches_hand_calculation() -> None:
    # Two strata, two households each. Worked by hand:
    # p = (2*1 + 1*1) / (2 + 2 + 1 + 1) = 0.5
    # z = w * (y - p) / 6 -> stratum A: [1/6, -1/6], stratum B: [1/12, -1/12]
    # V = 2 * (2/36) + 2 * (2/144) = 5/36
    design = pl.DataFrame(
        {
            "stratum_id": ["A", "A", "B", "B"],
            "hhid": [1, 2, 3, 4],
            "w": [2.0, 2.0, 1.0, 1.0],
            "yes": [1, 0, 1, 0],
            "valid": [1, 1, 1, 1],
        }
    )
    estimate = weighted_proportion(design, "yes", "valid", "w")
    assert [estimate.value, round(estimate.se, 12)] == [
        0.5,
        round(math.sqrt(5 / 36), 12),
    ]


def test_weighted_total_matches_hand_calculation() -> None:
    # T = 2 + 1 = 3. z = w*y -> A: [2, 0], B: [1, 0]. V = 2*2 + 2*0.5 = 5.
    design = pl.DataFrame(
        {
            "stratum_id": ["A", "A", "B", "B"],
            "hhid": [1, 2, 3, 4],
            "w": [2.0, 2.0, 1.0, 1.0],
            "yes": [1, 0, 1, 0],
        }
    )
    estimate = weighted_total(design, "yes", "w")
    assert [estimate.value, round(estimate.se, 12)] == [3.0, round(math.sqrt(5), 12)]


def test_rows_outside_the_domain_do_not_move_the_proportion() -> None:
    inside = pl.DataFrame(
        {
            "stratum_id": ["A"] * 3,
            "hhid": [1, 2, 3],
            "w": [1.0] * 3,
            "yes": [1, 0, 0],
            "valid": [1] * 3,
        }
    )
    outside = pl.DataFrame(
        {
            "stratum_id": ["A"] * 2,
            "hhid": [4, 5],
            "w": [9.0] * 2,
            "yes": [0, 0],
            "valid": [0, 0],
        }
    )
    alone = weighted_proportion(inside, "yes", "valid", "w")
    together = weighted_proportion(pl.concat([inside, outside]), "yes", "valid", "w")
    assert together.value == alone.value


def test_do_labels_are_read_for_one_variable(tmp_path: Path) -> None:
    do_file = tmp_path / "nsch_2020_topical.do"
    do_file.write_text(
        'label var k2q35a  "Autism ASD"\n'
        'label define k2q35a_lab  1  "Yes"\n'
        '    label define k2q35a_lab  2  "No", add\n'
        'label define k2q35a_lab  .m "No valid response", add\n'
        'label define k2q35a_1_years_lab  .m "No valid response"\n'
    )
    assert read_do_labels(do_file, "k2q35a") == {
        "variable_label": "Autism ASD",
        "1": "Yes",
        "2": "No",
        ".m": "No valid response",
    }


def test_two_candidate_files_for_one_year_are_refused(tmp_path: Path) -> None:
    (tmp_path / "nsch_2023_topical.dta").touch()
    (tmp_path / "nsch_2023e_topical.dta").touch()
    with pytest.raises(FileNotFoundError, match="found 2"):
        find_topical_file(tmp_path, 2023)
