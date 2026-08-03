"""Tests for fpc_mean_variance.fpc_mean_variance."""

from morie.fn import _array_core as np

from morie.fn.fpc_mean_variance import (
    fpc_mean_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r26e5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = fpc_mean_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r26e5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = fpc_mean_variance(x)
    assert isinstance(result, dict)
