"""Tests for mean_semivariance_stsi_variance.mean_semivariance_stsi_variance."""

from morie.fn import _array_core as np

from morie.fn.mean_semivariance_stsi_variance import (
    mean_semivariance_stsi_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = mean_semivariance_stsi_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = mean_semivariance_stsi_variance(x)
    assert isinstance(result, dict)
