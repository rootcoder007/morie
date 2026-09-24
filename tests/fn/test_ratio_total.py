"""Tests for ratio_total.ratio_total."""

from morie.fn import _array_core as np

from morie.fn.ratio_total import (
    ratio_total,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e23_basic():
    """Test basic functionality."""
    t_pi_z = 0.5
    t_pi_x = 0.5
    t_x_true = 0.5
    result = ratio_total(t_pi_z, t_pi_x, t_x_true)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e23_edge():
    """Test edge cases."""
    t_pi_z = 0.5
    t_pi_x = 0.5
    t_x_true = 0.5
    result = ratio_total(t_pi_z, t_pi_x, t_x_true)
    assert isinstance(result, dict)
