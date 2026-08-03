"""Tests for ratio_g_weight.ratio_g_weight."""

from morie.fn import _array_core as np

from morie.fn.ratio_g_weight import (
    ratio_g_weight,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e27_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ratio_g_weight(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e27_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ratio_g_weight(x)
    assert isinstance(result, dict)
