"""Tests for true_confidence_level.true_confidence_level."""

from morie.fn import _array_core as np

from morie.fn.true_confidence_level import (
    true_confidence_level,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e6_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = true_confidence_level(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e6_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = true_confidence_level(x)
    assert isinstance(result, dict)
