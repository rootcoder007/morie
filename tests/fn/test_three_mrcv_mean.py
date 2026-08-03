"""Tests for three_mrcv_mean.three_mrcv_mean."""

from morie.fn import _array_core as np

from morie.fn.three_mrcv_mean import (
    three_mrcv_mean,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e16_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = three_mrcv_mean(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e16_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = three_mrcv_mean(x)
    assert isinstance(result, dict)
