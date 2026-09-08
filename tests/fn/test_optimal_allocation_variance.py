"""Tests for optimal_allocation_variance.optimal_allocation_variance."""

from morie.fn import _array_core as np

from morie.fn.optimal_allocation_variance import (
    optimal_allocation_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e10_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = optimal_allocation_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e10_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = optimal_allocation_variance(x)
    assert isinstance(result, dict)
