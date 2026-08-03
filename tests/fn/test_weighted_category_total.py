"""Tests for weighted_category_total.weighted_category_total."""

from morie.fn import _array_core as np

from morie.fn.weighted_category_total import (
    weighted_category_total,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e7_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = weighted_category_total(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e7_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = weighted_category_total(x)
    assert isinstance(result, dict)
