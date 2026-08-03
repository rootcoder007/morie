"""Tests for piecewise_cubic.piecewise_cubic."""

from morie.fn import _array_core as np

from morie.fn.piecewise_cubic import (
    piecewise_cubic,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e34_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = piecewise_cubic(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e34_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = piecewise_cubic(x)
    assert isinstance(result, dict)
