"""Tests for ratio_g_weight.ratio_g_weight."""

from morie.fn import _array_core as np

from morie.fn.ratio_g_weight import (
    ratio_g_weight,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e27_basic():
    """Test basic functionality."""
    t_x_true = 0.5
    t_pi_x = 0.5
    result = ratio_g_weight(t_x_true, t_pi_x)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e27_edge():
    """Test edge cases."""
    t_x_true = 0.5
    t_pi_x = 0.5
    result = ratio_g_weight(t_x_true, t_pi_x)
    assert isinstance(result, dict)
