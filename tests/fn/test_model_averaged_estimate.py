"""Tests for model_averaged_estimate.model_averaged_estimate."""

from morie.fn import _array_core as np

from morie.fn.model_averaged_estimate import (
    model_averaged_estimate,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo5e3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = model_averaged_estimate(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo5e3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = model_averaged_estimate(x)
    assert isinstance(result, dict)
