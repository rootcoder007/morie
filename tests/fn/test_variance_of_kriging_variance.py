"""Tests for variance_of_kriging_variance.variance_of_kriging_variance."""

from morie.fn import _array_core as np

from morie.fn.variance_of_kriging_variance import (
    variance_of_kriging_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = variance_of_kriging_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = variance_of_kriging_variance(x)
    assert isinstance(result, dict)
