"""Tests for rpl_rt — region trend."""

from morie.fn import _frame_core as pd

from morie.fn.rpl_rt import rplace_region_trend


def test_rpl_rt_basic(otis_df):
    result = rplace_region_trend(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from morie.fn.rpl_rt import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
