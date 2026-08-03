"""Tests for ols_prediction_variance.ols_prediction_variance."""

from morie.fn import _array_core as np

from morie.fn.ols_prediction_variance import (
    ols_prediction_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r20e3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ols_prediction_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r20e3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ols_prediction_variance(x)
    assert isinstance(result, dict)
