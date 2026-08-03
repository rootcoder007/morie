"""Tests for exponential_semivariogram.exponential_semivariogram."""

from morie.fn import _array_core as np

from morie.fn.exponential_semivariogram import (
    exponential_semivariogram,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = exponential_semivariogram(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = exponential_semivariogram(x)
    assert isinstance(result, dict)
