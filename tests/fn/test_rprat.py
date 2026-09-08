"""Tests for rprat — placement rate."""

from morie.fn import _frame_core as pd

from morie.fn.rprat import rplace_rate


def test_rprat_basic(otis_df):
    result = rplace_rate(otis_df, population=100000)
    assert isinstance(result, pd.DataFrame)
    assert "rate_per_100k" in result.columns


def test_cheatsheet():
    from morie.fn.rprat import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
