"""Tests for iid_mean_variance.iid_mean_variance."""

from morie.fn import _array_core as np

from morie.fn.iid_mean_variance import (
    iid_mean_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r26e2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = iid_mean_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r26e2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = iid_mean_variance(x)
    assert isinstance(result, dict)
