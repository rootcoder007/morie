"""Tests for ols_prediction_variance.ols_prediction_variance."""

from morie.fn import _array_core as np

from morie.fn.ols_prediction_variance import (
    ols_prediction_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r20e3_basic():
    """Test basic functionality."""
    sigma2_eps = 0.5
    x0 = 0.5
    x = 0.5
    result = ols_prediction_variance(sigma2_eps, x0, x)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r20e3_edge():
    """Test edge cases."""
    sigma2_eps = 0.5
    x0 = 0.5
    x = 0.5
    result = ols_prediction_variance(sigma2_eps, x0, x)
    assert isinstance(result, dict)
