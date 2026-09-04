# SPDX-License-Identifier: AGPL-3.0-or-later
"""Victorian crime data loaders and analyses (no network)."""

import io
import math

import pytest

from morie.fn import _frame_core as pd
from morie.datasets_vic import (
    vic_catalog,
    vic_indigenous_ratio,
    vic_lga_rates,
    vic_offence_trend,
    vic_sheets,
    vic_table,
)


def test_catalog_lists_every_workbook():
    cat = vic_catalog()
    assert len(cat) == 27
    keys = [e["key"] for e in cat]
    assert len(set(keys)) == len(keys)
    assert all(e["url"].startswith("https://") for e in cat)
    assert all(e["file"].endswith(".xlsx") for e in cat)
    for k in ("criminal_incidents", "lga_criminal_incidents",
              "indigenous_victim_reports"):
        assert k in keys


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown key"):
        vic_table("no_such_table", cache_dir=str(tmp_path))
    with pytest.raises(ValueError, match="unknown key"):
        vic_sheets("no_such_table", cache_dir=str(tmp_path))


def test_loader_stays_offline(tmp_path):
    d = vic_table("criminal_incidents", 1, cache_dir=str(tmp_path))
    assert d.shape[0] == 0
    assert vic_sheets("criminal_incidents", cache_dir=str(tmp_path)) == []


def test_offence_trend_reports_change_per_division():
    d = pd.DataFrame({
        "Year": [2024, 2025, 2024, 2025],
        "Offence Division": ["A", "A", "B", "B"],
        "Incidents Recorded": [100, 150, 80, 60],
    })
    t = vic_offence_trend(d)
    rows = dict(zip(list(t["division"]), zip(list(t["abs_change"]),
                                             list(t["pct_change"]))))
    assert rows["A"] == (50, 50)          # 100 -> 150
    assert rows["B"] == (-20, -25)        # 80 -> 60


def test_lga_rates_rank_by_rate():
    d = pd.DataFrame({
        "Year": [2025, 2025, 2025],
        "Local Government Area": ["Alpha", "Beta", "Gamma"],
        "Incidents Recorded": [500, 300, 900],
        "Rate per 100,000 population": [1200, 800, 2500],
    })
    r = vic_lga_rates(d, top=2)
    assert list(r["lga"]) == ["Gamma", "Alpha"]
    assert list(r["rank"]) == [1, 2]


def test_indigenous_ratio_excludes_total_people():
    # The published table carries a third "Total People" status; folding
    # it into the non-Indigenous count roughly doubles the denominator.
    d = pd.DataFrame({
        "Year": [2026, 2026, 2026],
        "Offence Division": ["A", "A", "A"],
        "Indigenous Status": [
            "Aboriginal and/or Torres Strait Islander",
            "Non-Indigenous",
            "Total People",
        ],
        "Victim Reports": [10, 90, 100],
    })
    r = vic_indigenous_ratio(d)
    assert list(r["indigenous"]) == [10]
    assert list(r["non_indigenous"]) == [90]      # NOT 190
    assert abs(list(r["count_ratio"])[0] - 10 / 90) < 1e-12


def test_indigenous_ratio_uses_one_year():
    d = pd.DataFrame({
        "Year": [2025, 2025, 2026, 2026],
        "Offence Division": ["A", "A", "A", "A"],
        "Indigenous Status": [
            "Aboriginal and/or Torres Strait Islander", "Non-Indigenous",
            "Aboriginal and/or Torres Strait Islander", "Non-Indigenous",
        ],
        "Victim Reports": [1, 9, 20, 80],
    })
    assert list(vic_indigenous_ratio(d)["indigenous"]) == [20]
    assert list(vic_indigenous_ratio(d, year=2025)["indigenous"]) == [1]


def test_sheet_map_is_not_resolved_by_sorting_part_names():
    # "sheet10.xml" sorts BEFORE "sheet2.xml", so a workbook with ten or
    # more sheets used to resolve every sheet after the ninth to the
    # wrong part. Build such a workbook and read the tenth back.
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        for i in range(1, 12):
            frame = pd.DataFrame({"v": ["sheet%d" % i]})
            frame.to_excel(w, sheet_name="S%d" % i, index=False)
    buf.seek(0)
    xl = pd.ExcelFile(buf)
    assert xl.sheet_names == ["S%d" % i for i in range(1, 12)]
    buf.seek(0)
    tenth = pd.read_excel(buf, sheet_name="S10")
    assert list(tenth["v"]) == ["sheet10"]
    buf.seek(0)
    second = pd.read_excel(buf, sheet_name="S2")
    assert list(second["v"]) == ["sheet2"]
