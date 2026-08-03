"""Tests for gls_population_slope.gls_population_slope."""

from morie.fn import _array_core as np

from morie.fn.gls_population_slope import (
    gls_population_slope,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gls_population_slope(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gls_population_slope(x)
    assert isinstance(result, dict)
