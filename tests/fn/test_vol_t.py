"""Tests for vol_t — volatility trend."""

from morie.fn import _frame_core as pd

from morie.fn.vol_t import vol_trd


def test_vol_t_basic(otis_df):
    result = vol_trd(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from morie.fn.vol_t import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
