"""Tests for rpl_a — placement by age."""

from morie.fn import _frame_core as pd

from morie.fn.rpl_a import rplace_by_age


def test_rpl_a_basic(otis_df):
    result = rplace_by_age(otis_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_cheatsheet():
    from morie.fn.rpl_a import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
