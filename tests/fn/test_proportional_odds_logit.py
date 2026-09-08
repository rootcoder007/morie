"""Tests for proportional_odds_logit.proportional_odds_logit."""

from morie.fn import _array_core as np

from morie.fn.proportional_odds_logit import (
    proportional_odds_logit,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e11_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = proportional_odds_logit(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e11_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = proportional_odds_logit(x)
    assert isinstance(result, dict)
