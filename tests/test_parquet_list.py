# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native Parquet reader: three-level LIST columns (the A2AJ citation fields)."""

from pathlib import Path

import pytest

from morie.fn._parquet_core import read_parquet

FIXTURES = Path(__file__).parent / "fixtures" / "parquet"
EXPECTED = [["2020 SCC 5", "2019 ONCA 1"], [], None, ["x", None], ["2001 FC 3"]]


@pytest.mark.parametrize("name", ["list_snappy", "list_dict", "list_plain"])
def test_list_column_round_trips_null_empty_and_null_element(name):
    df = read_parquet(str(FIXTURES / f"{name}.parquet"))
    assert list(df.columns) == ["id", "name", "cites"]
    assert list(df["id"]) == [1, 2, 3, 4, 5]
    assert list(df["name"]) == ["a", "b", None, "d", "e"]
    assert list(df["cites"]) == EXPECTED


def test_list_column_survives_column_selection():
    df = read_parquet(str(FIXTURES / "list_snappy.parquet"), columns=["cites", "id"])
    assert list(df.columns) == ["cites", "id"]
    assert list(df["cites"]) == EXPECTED
