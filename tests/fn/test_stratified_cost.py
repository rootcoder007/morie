"""Tests for stratified_cost.stratified_cost."""

from morie.fn import _array_core as np

from morie.fn.stratified_cost import (
    stratified_cost,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r4e18_basic():
    """Test basic functionality."""
    c0 = 0.5
    stratum_costs = 0.5
    stratum_sizes = 0.5
    result = stratified_cost(c0, stratum_costs, stratum_sizes)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r4e18_edge():
    """Test edge cases."""
    c0 = 0.5
    stratum_costs = 0.5
    stratum_sizes = 0.5
    result = stratified_cost(c0, stratum_costs, stratum_sizes)
    assert isinstance(result, dict)
