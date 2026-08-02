"""Tests for rpl_r — placement by region."""

from morie.fn import _frame_core as pd

from morie.fn.rpl_r import rplace_by_region


def test_rpl_r_basic(otis_df):
    result = rplace_by_region(otis_df, "Central")
    assert isinstance(result, pd.DataFrame)
    assert "year" in result.columns
    assert "n_individuals" in result.columns


def test_rpl_r_empty_region(otis_df):
    result = rplace_by_region(otis_df, "Nonexistent")
    assert len(result) == 0
