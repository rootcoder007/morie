"""Tests for ht_mean.ht_mean."""

from morie.fn import _array_core as np

from morie.fn.ht_mean import (
    ht_mean,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r2e4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ht_mean(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r2e4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ht_mean(x)
    assert isinstance(result, dict)
