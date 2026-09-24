"""Tests for autocorrelated_mean_variance.autocorrelated_mean_variance."""

import math

from morie.fn import _array_core as np

from morie.fn.autocorrelated_mean_variance import (
    autocorrelated_mean_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r26e3_basic():
    """Test basic functionality."""
    sigma2 = 1.0
    n = 40
    rho_bar = 0.3
    result = autocorrelated_mean_variance(sigma2, n, rho_bar)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    expected = (sigma2 / n) * (1 + (n - 1) * rho_bar)
    assert math.isclose(result["value"], expected)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r26e3_edge():
    """Test edge case: n=1 so autocorrelation has no effect."""
    sigma2 = 2.5
    n = 1
    rho_bar = 0.7
    result = autocorrelated_mean_variance(sigma2, n, rho_bar)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    expected = sigma2
    assert math.isclose(result["value"], expected)
