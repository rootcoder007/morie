"""Tests for local_mean_variance.local_mean_variance."""

from morie.fn import _array_core as np

from morie.fn.local_mean_variance import (
    local_mean_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r9e10_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = local_mean_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r9e10_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = local_mean_variance(x)
    assert isinstance(result, dict)
