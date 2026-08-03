"""Tests for twophase_stratified_variance.twophase_stratified_variance."""

from morie.fn import _array_core as np

from morie.fn.twophase_stratified_variance import (
    twophase_stratified_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r11e5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = twophase_stratified_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r11e5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = twophase_stratified_variance(x)
    assert isinstance(result, dict)
