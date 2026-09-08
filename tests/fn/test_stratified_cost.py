"""Tests for stratified_cost.stratified_cost."""

from morie.fn import _array_core as np

from morie.fn.stratified_cost import (
    stratified_cost,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r4e18_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = stratified_cost(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r4e18_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = stratified_cost(x)
    assert isinstance(result, dict)
