"""Tests for twophase_stratified_variance.twophase_stratified_variance."""

from morie.fn import _array_core as np

from morie.fn.twophase_stratified_variance import (
    twophase_stratified_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r11e5_basic():
    """Test basic functionality."""
    n1h = 0.5
    n1 = 0.5
    s2_2h = 0.5
    n2h = 0.5
    zbar_2h = 0.5
    zbar_hat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = twophase_stratified_variance(n1h, n1, s2_2h, n2h, zbar_2h, zbar_hat)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r11e5_edge():
    """Test edge cases."""
    n1h = 0.5
    n1 = 0.5
    s2_2h = 0.5
    n2h = 0.5
    zbar_2h = 0.5
    zbar_hat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = twophase_stratified_variance(n1h, n1, s2_2h, n2h, zbar_2h, zbar_hat)
    assert isinstance(result, dict)
