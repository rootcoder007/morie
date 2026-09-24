"""Tests for gls_population_slope.gls_population_slope."""

from morie.fn import _array_core as np

from morie.fn.gls_population_slope import (
    gls_population_slope,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e4_basic():
    """Test basic functionality."""
    x = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    z = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    sigma2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = gls_population_slope(x, z, sigma2)
    assert isinstance(result, dict)
    assert "values" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e4_edge():
    """Test edge cases."""
    x = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    z = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    sigma2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = gls_population_slope(x, z, sigma2)
    assert isinstance(result, dict)
