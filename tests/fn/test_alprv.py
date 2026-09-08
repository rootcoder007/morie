"""Tests for alprv — alert prevalence by group."""

from morie.fn import _frame_core as pd

from morie.fn.alprv import alprev


def test_alprv_basic(otis_df):
    result = alprev(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from morie.fn.alprv import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
