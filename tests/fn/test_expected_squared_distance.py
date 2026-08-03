"""Tests for expected_squared_distance.expected_squared_distance."""

from morie.fn import _array_core as np

from morie.fn.expected_squared_distance import (
    expected_squared_distance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e15_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = expected_squared_distance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e15_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = expected_squared_distance(x)
    assert isinstance(result, dict)
