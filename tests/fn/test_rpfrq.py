"""Tests for rpfrq — repeat placement frequency."""

from morie.fn import _frame_core as pd

from morie.fn.rpfrq import rplace_frequency


def test_rpfrq_basic(otis_df):
    result = rplace_frequency(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from morie.fn.rpfrq import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
