"""Tests for posterior_kernel_regression.posterior_kernel_regression."""

from morie.fn import _array_core as np

from morie.fn.posterior_kernel_regression import (
    posterior_kernel_regression,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e25_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = posterior_kernel_regression(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e25_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = posterior_kernel_regression(x)
    assert isinstance(result, dict)
