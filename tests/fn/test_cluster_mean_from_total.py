"""Tests for cluster_mean_from_total.cluster_mean_from_total."""

from morie.fn import _array_core as np

from morie.fn.cluster_mean_from_total import (
    cluster_mean_from_total,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r6e10_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cluster_mean_from_total(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r6e10_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cluster_mean_from_total(x)
    assert isinstance(result, dict)
