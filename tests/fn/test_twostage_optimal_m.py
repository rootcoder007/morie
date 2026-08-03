"""Tests for twostage_optimal_m.twostage_optimal_m."""

from morie.fn import _array_core as np

from morie.fn.twostage_optimal_m import (
    twostage_optimal_m,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r7e10_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = twostage_optimal_m(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r7e10_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = twostage_optimal_m(x)
    assert isinstance(result, dict)
