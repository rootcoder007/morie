"""Tests for si_proportion.si_proportion."""

from morie.fn import _array_core as np

from morie.fn.si_proportion import (
    si_proportion,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e6_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = si_proportion(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e6_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = si_proportion(x)
    assert isinstance(result, dict)
