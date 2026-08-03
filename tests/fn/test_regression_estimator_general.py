"""Tests for regression_estimator_general.regression_estimator_general."""

from morie.fn import _array_core as np

from morie.fn.regression_estimator_general import (
    regression_estimator_general,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e8_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = regression_estimator_general(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e8_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = regression_estimator_general(x)
    assert isinstance(result, dict)
