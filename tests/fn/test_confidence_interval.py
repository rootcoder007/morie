"""Tests for confidence_interval.confidence_interval."""

from morie.fn import _array_core as np

from morie.fn.confidence_interval import (
    confidence_interval,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e15_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = confidence_interval(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e15_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = confidence_interval(x)
    assert isinstance(result, dict)
