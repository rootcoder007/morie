"""Tests for truncated_power_spline.truncated_power_spline."""

from morie.fn import _array_core as np

from morie.fn.truncated_power_spline import (
    truncated_power_spline,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e36_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = truncated_power_spline(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e36_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = truncated_power_spline(x)
    assert isinstance(result, dict)
