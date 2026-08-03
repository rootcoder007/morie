"""Tests for regression_total.regression_total."""

from morie.fn import _array_core as np

from morie.fn.regression_total import (
    regression_total,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r9e2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = regression_total(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r9e2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = regression_total(x)
    assert isinstance(result, dict)
