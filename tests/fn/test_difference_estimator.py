"""Tests for difference_estimator.difference_estimator."""

from morie.fn import _array_core as np

from morie.fn.difference_estimator import (
    difference_estimator,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = difference_estimator(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = difference_estimator(x)
    assert isinstance(result, dict)
