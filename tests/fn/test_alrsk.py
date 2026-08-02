"""Tests for alrsk — alert risk score."""

from morie.fn import _frame_core as pd

from morie.fn.alrsk import alrisk


def test_alrsk_basic(otis_df):
    result = alrisk(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from morie.fn.alrsk import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
