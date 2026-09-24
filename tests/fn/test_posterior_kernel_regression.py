"""Tests for posterior_kernel_regression.posterior_kernel_regression."""

from morie.fn import _array_core as np

from morie.fn.posterior_kernel_regression import (
    posterior_kernel_regression,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e25_basic():
    """Test basic functionality."""
    logliks = np.random.default_rng(42).normal(0.0, 1.0, 40)
    log_priors = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = posterior_kernel_regression(logliks, log_priors)
    assert isinstance(result, dict)
    assert "value" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e25_edge():
    """Test edge cases."""
    logliks = np.random.default_rng(42).normal(0.0, 1.0, 40)
    log_priors = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = posterior_kernel_regression(logliks, log_priors)
    assert isinstance(result, dict)
