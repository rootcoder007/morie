"""Tests for optimal_allocation_variance.optimal_allocation_variance."""

from morie.fn import _array_core as np

from morie.fn.optimal_allocation_variance import (
    optimal_allocation_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e10_basic():
    """Test basic functionality."""
    weights = 0.5
    s_h = 0.5
    c_h = 0.5
    n = 0.5
    result = optimal_allocation_variance(weights, s_h, c_h, n)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e10_edge():
    """Test edge cases."""
    weights = 0.5
    s_h = 0.5
    c_h = 0.5
    n = 0.5
    result = optimal_allocation_variance(weights, s_h, c_h, n)
    assert isinstance(result, dict)
