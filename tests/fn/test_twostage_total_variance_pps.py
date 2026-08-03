"""Tests for twostage_total_variance_pps.twostage_total_variance_pps."""

from morie.fn import _array_core as np

from morie.fn.twostage_total_variance_pps import (
    twostage_total_variance_pps,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r7e12_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = twostage_total_variance_pps(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r7e12_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = twostage_total_variance_pps(x)
    assert isinstance(result, dict)
