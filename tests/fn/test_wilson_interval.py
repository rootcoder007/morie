"""Tests for wilson_interval.wilson_interval."""

from morie.fn import _array_core as np

from morie.fn.wilson_interval import (
    wilson_interval,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilson_interval(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilson_interval(x)
    assert isinstance(result, dict)
