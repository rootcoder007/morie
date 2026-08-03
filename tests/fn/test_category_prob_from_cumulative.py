"""Tests for category_prob_from_cumulative.category_prob_from_cumulative."""

from morie.fn import _array_core as np

from morie.fn.category_prob_from_cumulative import (
    category_prob_from_cumulative,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e12_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = category_prob_from_cumulative(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e12_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = category_prob_from_cumulative(x)
    assert isinstance(result, dict)
