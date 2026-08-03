"""Tests for cluster_total_pps.cluster_total_pps."""

from morie.fn import _array_core as np

from morie.fn.cluster_total_pps import (
    cluster_total_pps,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r6e4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cluster_total_pps(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r6e4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cluster_total_pps(x)
    assert isinstance(result, dict)
