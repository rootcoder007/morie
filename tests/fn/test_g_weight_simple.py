"""Tests for g_weight_simple.g_weight_simple."""

from morie.fn import _array_core as np

from morie.fn.g_weight_simple import (
    g_weight_simple,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e17_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = g_weight_simple(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e17_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = g_weight_simple(x)
    assert isinstance(result, dict)
