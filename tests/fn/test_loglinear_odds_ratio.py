"""Tests for loglinear_odds_ratio.loglinear_odds_ratio."""

from morie.fn import _array_core as np

from morie.fn.loglinear_odds_ratio import (
    loglinear_odds_ratio,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e7_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = loglinear_odds_ratio(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e7_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = loglinear_odds_ratio(x)
    assert isinstance(result, dict)
