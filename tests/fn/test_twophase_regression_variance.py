"""Tests for twophase_regression_variance.twophase_regression_variance."""

from morie.fn import _array_core as np

from morie.fn.twophase_regression_variance import (
    twophase_regression_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r11e7_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = twophase_regression_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r11e7_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = twophase_regression_variance(x)
    assert isinstance(result, dict)
