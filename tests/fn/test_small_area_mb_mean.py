"""Tests for small_area_mb_mean.small_area_mb_mean."""

from morie.fn import _array_core as np

from morie.fn.small_area_mb_mean import (
    small_area_mb_mean,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r14e15_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = small_area_mb_mean(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r14e15_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = small_area_mb_mean(x)
    assert isinstance(result, dict)
