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
    TOPICAL_COLUMNS,
    add_indicators,
    classify_autism,
    find_topical_file,
    load_children,
    prevalence_table,
    read_do_labels,
    read_topical_year,
    sha256_of,
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


# --- the file reader, against real .dta files written into tmp_path ---------


def _write_topical(path: Path, rows: dict[str, list[object]]) -> None:
    """Write a tiny Census-shaped topical file, tagged missings included."""
    import pandas as pd
    import pyreadstat

    pyreadstat.write_dta(
        pd.DataFrame(rows),
        str(path),
        missing_user_values={"k2q35a": ["m"], "k2q35b": ["l", "m"]},
    )


def _topical_rows(year: int, hhids: list[int]) -> dict[str, list[object]]:
    # Lists are invariant, so the mixed-type columns are declared as
    # list[object] before being sliced; a sliced str literal stays list[str].
    n = len(hhids)
    stratum: list[object] = ["1", "2A", "2A", "1"]
    ages: list[object] = [5, 9, 10, 2]
    ever: list[object] = [1.0, 2.0, "m", 1.0]
    current: list[object] = [1.0, "l", "m", float("nan")]
    return {
        "year": [year] * n,
        "hhid": list(hhids),
        "fipsst": [4] * n,
        "stratum": stratum[:n],
        "fwc": [1.0] * n,
        "sc_age_years": ages[:n],
        "k2q35a": ever[:n],
        "k2q35b": current[:n],
    }


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    target = tmp_path / "blob.bin"
    target.write_bytes(b"nsch" * 1000)
    assert sha256_of(target) == hashlib.sha256(b"nsch" * 1000).hexdigest()


def test_topical_file_is_read_as_string_codes_with_stratum_recoded(tmp_path: Path) -> None:
    path = tmp_path / "nsch_2020e_topical.dta"
    _write_topical(path, _topical_rows(2020, [1, 2, 3, 4]))
    result = read_topical_year(path, 2020)
    assert result.columns == list(TOPICAL_COLUMNS)
    assert result["year"].to_list() == [2020, 2020, 2020, 2020]
    assert result["stratum"].to_list() == ["1", "2", "2", "1"]
    assert result["k2q35a"].to_list() == ["1", "2", "m", "1"]
    # A system missing value must not look like an answer or a tagged skip.
    assert result["k2q35b"].to_list() == ["1", "l", "m", "sysmis"]
    assert result["fwc"].dtype == pl.Float64


def test_file_whose_year_column_disagrees_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "nsch_2021e_topical.dta"
    _write_topical(path, _topical_rows(2020, [1, 2, 3]))
    with pytest.raises(ValueError, match="expected year 2021"):
        read_topical_year(path, 2021)


def test_file_missing_a_required_column_is_refused(tmp_path: Path) -> None:
    import pandas as pd
    import pyreadstat

    path = tmp_path / "nsch_2020e_topical.dta"
    rows = _topical_rows(2020, [1, 2, 3])
    del rows["fwc"]
    pyreadstat.write_dta(pd.DataFrame(rows), str(path))
    with pytest.raises(KeyError, match="fwc"):
        read_topical_year(path, 2020)


def test_prevalence_table_end_to_end_matches_hand_counts(tmp_path: Path) -> None:
    # 2020, weight 1: one child ever and currently, one never, one who did not
    # answer, and a two-year-old outside the age band. 2021, weight 2: two
    # ever, of whom one currently, and two never. Pooling halves each weight.
    rows_2020: dict[str, list[object]] = {
        "year": [2020] * 4,
        "hhid": [1, 2, 3, 4],
        "fipsst": [4] * 4,
        "stratum": ["1", "2A", "2A", "1"],
        "fwc": [1.0] * 4,
        "sc_age_years": [5, 9, 10, 2],
        "k2q35a": [1.0, 2.0, "m", 1.0],
        "k2q35b": [1.0, "l", "m", 1.0],
    }
    rows_2021: dict[str, list[object]] = {
        "year": [2021] * 4,
        "hhid": [5, 6, 7, 8],
        "fipsst": [4] * 4,
        "stratum": ["1", "1", "2A", "2A"],
        "fwc": [2.0] * 4,
        "sc_age_years": [6, 7, 8, 16],
        "k2q35a": [1.0, 1.0, 2.0, 2.0],
        "k2q35b": [1.0, 2.0, "l", "l"],
    }
    (tmp_path / "2020").mkdir()
    (tmp_path / "2021").mkdir()
    _write_topical(tmp_path / "2020" / "nsch_2020e_topical.dta", rows_2020)
    _write_topical(tmp_path / "2021" / "nsch_2021e_topical.dta", rows_2021)

    children = load_children(tmp_path, [2020, 2021])
    table = prevalence_table(children, [2020, 2021])

    assert table["period"].to_list() == ["2020", "2021", "All years"]
    assert table["children_in_age_band"].to_list() == [3, 4, 7]
    assert table["ever_n"].to_list() == [1, 2, 3]
    assert table["ever_denominator"].to_list() == [2, 4, 6]
    assert table["current_n"].to_list() == [1, 1, 2]
    assert table["current_denominator"].to_list() == [2, 4, 6]
    # Weighted percent: 2020 is 1/2, 2021 is 4/8, pooled is (0.5 + 2) / (1 + 4).
    assert [round(v, 6) for v in table["ever_pct_weighted"].to_list()] == [50.0, 50.0, 50.0]
    # Weighted population: 2020 is 1, 2021 is 4, pooled is (1 + 4) / 2.
    assert table["ever_population"].to_list() == [1.0, 4.0, 2.5]
    assert table["current_population"].to_list() == [1.0, 2.0, 1.5]


def test_do_file_without_the_variable_is_refused(tmp_path: Path) -> None:
    do_file = tmp_path / "nsch_2020_topical.do"
    do_file.write_text('label var k2q35b  "Autism ASD Currently"\n')
    with pytest.raises(KeyError, match="k2q35a"):
        read_do_labels(do_file, "k2q35a")
