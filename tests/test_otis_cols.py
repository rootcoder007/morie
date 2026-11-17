"""Ontario renamed every OTIS column on 2026-10-30: the analyses read either spelling."""

import pytest

from morie import mrm_otis, otis, otis_all_analyze, otis_churn
from morie.fn import _frame_core as pd
from morie.otis_datasets import OtisColumnError, _otis_colmap, otis_cols


def respell(d, table):
    new = {o: n for t, o, n in _otis_colmap() if t == table}
    return d.rename(columns={c: new.get(c, c) for c in d.columns})


def b09_old():
    return pd.DataFrame(
        {
            "EndFiscalYear": [2025] * 4 + [2026] * 4,
            "NumberPlacements_Segregation": ["1", "2", "3 to 5", "Greater than 10"] * 2,
            "Gender": ["Male", "Female"] * 4,
            "NumberIndividuals_Segregation": [50, 20, 10, 3, 60, 25, 12, 4],
        }
    )


def b01_old(n=40):
    regions = ["Toronto", "Central", "Eastern", "Western"]
    return pd.DataFrame(
        {
            "EndFiscalYear": [2025 + i % 2 for i in range(n)],
            "UniqueIndividual_ID": [f"id{i % 15}" for i in range(n)],
            "Gender": ["Male", "Female"] * (n // 2),
            "Region_AtTimeOfPlacement": [regions[i % 4] for i in range(n)],
            "Region_MostRecentPlacement": [regions[(i // 3) % 4] for i in range(n)],
            "Age_Category": ["18 to 24", "25 to 49", "50+", "25 to 49"] * (n // 4),
            "NumberConsecutiveDays_Segregation": [1 + (i * 7) % 30 for i in range(n)],
            "MentalHealth_Alert": ["Yes", "No"] * (n // 2),
            "SuicideRisk_Alert": ["No", "No", "Yes", "No"] * (n // 4),
            "SuicideWatch_Alert": ["No"] * n,
            "Number_Of_Placements": [1 + i % 3 for i in range(n)],
        }
    )


def same(a, b):
    assert a.summary_lines == b.summary_lines
    assert a.tables == b.tables


def test_helper_resolves_exact_case_and_renamed_columns():
    d = pd.DataFrame(
        {
            "end_fiscal_year": [2026],
            "number_of_segregation_placements": ["1"],
            "GENDER": ["Male"],
            "NumberIndividuals_Segregation": [3],
        }
    )
    out = otis_cols(d, table="b09")
    assert list(out.columns) == [
        "EndFiscalYear",
        "NumberPlacements_Segregation",
        "Gender",
        "NumberIndividuals_Segregation",
    ]
    # the same new name means a different old one in b03: the expected name decides
    b03 = otis_cols(pd.DataFrame({"number_of_segregation_placements": [1]}), need=["Number_SegregationPlacements"])
    assert list(b03.columns) == ["Number_SegregationPlacements"]
    assert list(otis_cols(pd.DataFrame({"x": [1]}), table="b09").columns) == ["x"]


def test_two_columns_for_one_name_are_refused_and_a_missing_one_is_named():
    with pytest.raises(OtisColumnError, match="EndFiscalYear and end_fiscal_year all resolve to EndFiscalYear"):
        otis_cols(pd.DataFrame({"EndFiscalYear": [2025], "end_fiscal_year": [2026]}), table="b09")
    with pytest.raises(KeyError, match=r"missing column\(s\): EndFiscalYear; the table has: year, n"):
        otis_cols(pd.DataFrame({"year": [1], "n": [2]}), need=["EndFiscalYear"], table="b09")


def test_per_table_analyzers_give_the_same_result_on_either_spelling():
    same(otis_all_analyze.analyze_b09(respell(b09_old(), "b09")), otis_all_analyze.analyze_b09(b09_old()))
    d07 = pd.DataFrame(
        {
            "Year": [2025, 2025],
            "Alert_Type": ["Mental Health", "None"],
            "Housing_Type": ["Segregation", "General Population"],
            "Number_CustodialDeaths": [2, 5],
        }
    )
    r = otis_all_analyze.analyze_d07(respell(d07, "d07"))
    same(r, otis_all_analyze.analyze_d07(d07))
    assert r.tables[0]["rows"]
    with pytest.raises(OtisColumnError, match=r"the OTIS b09 table: missing column\(s\): EndFiscalYear"):
        otis_all_analyze.analyze_b09(pd.DataFrame({"a": [1], "b": [2]}))


def test_mandela_and_churn_read_ontarios_spelling():
    c11 = pd.DataFrame(
        {
            "EndFiscalYear": [2026] * 3,
            "Aggregate_Duration": ["1 to 15 days", "16 to 30 days", "Greater than 60 days"],
            "NumberIndividuals_RestrictiveConfinement": [100, 20, 5],
            "NumberIndividuals_Segregation": [80, 10, 2],
        }
    )
    same(
        otis_all_analyze.analyze_c11_mandela_classification(respell(c11, "c11")),
        otis_all_analyze.analyze_c11_mandela_classification(c11),
    )
    same(
        otis_churn.within_year_placement_count(respell(b01_old(), "b01")),
        otis_churn.within_year_placement_count(b01_old()),
    )


def test_mrm_callables_read_ontarios_spelling():
    a = mrm_otis.mrm_otis_placement_concentration(respell(b09_old(), "b09"))
    assert a.to_dict() == mrm_otis.mrm_otis_placement_concentration(b09_old()).to_dict()
    r = mrm_otis.mrm_otis_region_locality(respell(b01_old(), "b01"))
    assert r.chi2 == mrm_otis.mrm_otis_region_locality(b01_old()).chi2
    with pytest.raises(KeyError, match="NumberConsecutiveDays_Segregation"):
        mrm_otis.mrm_otis_seg_duration_km(pd.DataFrame({"x": [1]}))


def test_a01_primitives_read_either_spelling():
    old = b01_old()
    snake = old.rename(columns={c: n for t, c, n in _otis_colmap() if t == "a01"})
    assert otis.rctrnd(old).to_dict() == otis.rctrnd(snake).to_dict()
