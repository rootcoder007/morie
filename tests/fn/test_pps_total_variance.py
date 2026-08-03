"""Tests for pps_total_variance.pps_total_variance."""

from morie.fn import _array_core as np

from morie.fn.pps_total_variance import (
    pps_total_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r8e2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = pps_total_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r8e2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = pps_total_variance(x)
    assert isinstance(result, dict)
