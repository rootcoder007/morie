"""Tests for si_proportion_variance.si_proportion_variance."""

from morie.fn import _array_core as np

from morie.fn.si_proportion_variance import (
    si_proportion_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e14_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = si_proportion_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e14_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = si_proportion_variance(x)
    assert isinstance(result, dict)
