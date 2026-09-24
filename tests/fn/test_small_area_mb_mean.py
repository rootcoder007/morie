"""Tests for small_area_mb_mean.small_area_mb_mean."""

from morie.fn import _array_core as np

from morie.fn.small_area_mb_mean import (
    small_area_mb_mean,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r14e15_basic():
    """Test basic functionality."""
    xbar_d = 0.5
    beta_hat = 0.5
    v_d = 0.5
    result = small_area_mb_mean(xbar_d, beta_hat, v_d)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r14e15_edge():
    """Test edge cases."""
    xbar_d = 0.5
    beta_hat = 0.5
    v_d = 0.5
    result = small_area_mb_mean(xbar_d, beta_hat, v_d)
    assert isinstance(result, dict)
