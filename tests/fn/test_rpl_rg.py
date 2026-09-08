"""Tests for rpl_rg — region x gender cross-tab."""

from morie.fn import _frame_core as pd

from morie.fn.rpl_rg import rplace_region_gender


def test_rpl_rg_basic(otis_df):
    result = rplace_region_gender(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from morie.fn.rpl_rg import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
