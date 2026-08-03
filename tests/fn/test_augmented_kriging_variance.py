"""Tests for augmented_kriging_variance.augmented_kriging_variance."""

from morie.fn import _array_core as np

from morie.fn.augmented_kriging_variance import (
    augmented_kriging_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = augmented_kriging_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = augmented_kriging_variance(x)
    assert isinstance(result, dict)
