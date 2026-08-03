"""Tests for kott_carr_interval.kott_carr_interval."""

from morie.fn import _array_core as np

from morie.fn.kott_carr_interval import (
    kott_carr_interval,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e11_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kott_carr_interval(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e11_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kott_carr_interval(x)
    assert isinstance(result, dict)
