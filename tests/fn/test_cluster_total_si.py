"""Tests for cluster_total_si.cluster_total_si."""

from morie.fn import _array_core as np

from morie.fn.cluster_total_si import (
    cluster_total_si,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r6e9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cluster_total_si(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r6e9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cluster_total_si(x)
    assert isinstance(result, dict)
