"""Tests for rpgap — gap between placements."""

from morie.fn import _frame_core as pd

from morie.fn.rpgap import rplace_gap


def test_rpgap_basic(otis_df):
    result = rplace_gap(otis_df, date_col="start_date")
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from morie.fn.rpgap import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
