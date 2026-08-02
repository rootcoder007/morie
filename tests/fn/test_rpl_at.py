"""Tests for rpl_at — age trend."""

from morie.fn import _frame_core as pd

from morie.fn.rpl_at import rplace_age_trend


def test_rpl_at_basic(otis_df):
    result = rplace_age_trend(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from morie.fn.rpl_at import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
