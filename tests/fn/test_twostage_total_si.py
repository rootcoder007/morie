"""Tests for twostage_total_si.twostage_total_si."""

from morie.fn import _array_core as np

from morie.fn.twostage_total_si import (
    twostage_total_si,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r7e13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = twostage_total_si(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r7e13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = twostage_total_si(x)
    assert isinstance(result, dict)
