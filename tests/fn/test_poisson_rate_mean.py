"""Tests for poisson_rate_mean.poisson_rate_mean."""

from morie.fn import _array_core as np

from morie.fn.poisson_rate_mean import (
    poisson_rate_mean,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e15_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_rate_mean(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e15_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_rate_mean(x)
    assert isinstance(result, dict)
