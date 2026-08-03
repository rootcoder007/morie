"""Tests for linear_predictor_variance.linear_predictor_variance."""

from morie.fn import _array_core as np

from morie.fn.linear_predictor_variance import (
    linear_predictor_variance,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e16_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = linear_predictor_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e16_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = linear_predictor_variance(x)
    assert isinstance(result, dict)
