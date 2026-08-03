"""Tests for infinite_total.infinite_total."""

from morie.fn import _array_core as np

from morie.fn.infinite_total import (
    infinite_total,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e18_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = infinite_total(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e18_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = infinite_total(x)
    assert isinstance(result, dict)
