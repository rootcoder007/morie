"""Tests for ratio_total.ratio_total."""

from morie.fn import _array_core as np

from morie.fn.ratio_total import (
    ratio_total,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e23_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ratio_total(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e23_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ratio_total(x)
    assert isinstance(result, dict)
