"""Tests for rpl_t — placement trend."""

from morie.fn import _frame_core as pd

from morie.fn.rpl_t import rplace_trend


def test_rpl_t_basic(otis_df):
    result = rplace_trend(otis_df)
    assert isinstance(result, pd.DataFrame)
    assert "year" in result.columns


def test_cheatsheet():
    from morie.fn.rpl_t import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
