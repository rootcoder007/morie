"""Tests for loglinear_saturated_mean.loglinear_saturated_mean."""

from morie.fn import _array_core as np

from morie.fn.loglinear_saturated_mean import (
    loglinear_saturated_mean,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e6_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = loglinear_saturated_mean(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e6_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = loglinear_saturated_mean(x)
    assert isinstance(result, dict)
