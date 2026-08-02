"""Tests for vol_r — volatility by region."""

from morie.fn import _frame_core as pd

from morie.fn.vol_r import vol_reg


def test_vol_r_basic(otis_df):
    result = vol_reg(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from morie.fn.vol_r import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
