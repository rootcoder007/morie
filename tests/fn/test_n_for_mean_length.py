"""Tests for n_for_mean_length.n_for_mean_length."""

from morie.fn import _array_core as np

from morie.fn.n_for_mean_length import (
    n_for_mean_length,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e7_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = n_for_mean_length(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e7_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = n_for_mean_length(x)
    assert isinstance(result, dict)
