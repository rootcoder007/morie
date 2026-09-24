"""Tests for trend_weights.trend_weights."""

from morie.fn import _array_core as np

from morie.fn.trend_weights import (
    trend_weights,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r15e4_basic():
    """Test basic functionality."""
    times = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = trend_weights(times)
    assert isinstance(result, dict)
    assert "values" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r15e4_edge():
    """Test edge cases."""
    times = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = trend_weights(times)
    assert isinstance(result, dict)
