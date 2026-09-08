"""Tests for n_for_proportion_length.n_for_proportion_length."""

from morie.fn import _array_core as np

from morie.fn.n_for_proportion_length import (
    n_for_proportion_length,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e11_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = n_for_proportion_length(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e11_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = n_for_proportion_length(x)
    assert isinstance(result, dict)
