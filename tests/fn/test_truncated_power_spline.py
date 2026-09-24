"""Tests for truncated_power_spline.truncated_power_spline."""

from morie.fn import _array_core as np

from morie.fn.truncated_power_spline import (
    truncated_power_spline,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e36_basic():
    """Test basic functionality."""
    x = 0.5
    betas = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    knots = 0.5
    result = truncated_power_spline(x, betas, knots)
    assert isinstance(result, dict)
    assert "value" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e36_edge():
    """Test edge cases."""
    x = 0.5
    betas = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    knots = 0.5
    result = truncated_power_spline(x, betas, knots)
    assert isinstance(result, dict)
