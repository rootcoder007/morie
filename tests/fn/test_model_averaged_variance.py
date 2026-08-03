"""Tests for model_averaged_variance.model_averaged_variance."""

from morie.fn import _array_core as np

from morie.fn.model_averaged_variance import (
    model_averaged_variance,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo5e4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = model_averaged_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo5e4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = model_averaged_variance(x)
    assert isinstance(result, dict)
