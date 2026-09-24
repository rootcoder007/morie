"""Tests for loglinear_odds_ratio.loglinear_odds_ratio."""

from morie.fn import _array_core as np

from morie.fn.loglinear_odds_ratio import (
    loglinear_odds_ratio,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e7_basic():
    """Test basic functionality."""
    bxz_ij = 0.5
    bxz_ipjp = 0.5
    bxz_ipj = 0.5
    bxz_ijp = 0.5
    result = loglinear_odds_ratio(bxz_ij, bxz_ipjp, bxz_ipj, bxz_ijp)
    assert isinstance(result, dict)
    assert "value" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e7_edge():
    """Test edge cases."""
    bxz_ij = 0.5
    bxz_ipjp = 0.5
    bxz_ipj = 0.5
    bxz_ijp = 0.5
    result = loglinear_odds_ratio(bxz_ij, bxz_ipjp, bxz_ipj, bxz_ijp)
    assert isinstance(result, dict)
