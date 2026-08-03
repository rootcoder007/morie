"""Tests for nonproportional_odds_logit.nonproportional_odds_logit."""

from morie.fn import _array_core as np

from morie.fn.nonproportional_odds_logit import (
    nonproportional_odds_logit,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e16_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = nonproportional_odds_logit(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e16_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = nonproportional_odds_logit(x)
    assert isinstance(result, dict)
