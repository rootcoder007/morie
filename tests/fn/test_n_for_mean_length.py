"""Tests for n_for_mean_length.n_for_mean_length."""

from morie.fn import _array_core as np

from morie.fn.n_for_mean_length import (
    n_for_mean_length,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e7_basic():
    """Test basic functionality."""
    u_crit = 0.5
    s_star = 0.5
    l_max = 0.5
    result = n_for_mean_length(u_crit, s_star, l_max)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e7_edge():
    """Test edge cases."""
    u_crit = 0.5
    s_star = 0.5
    l_max = 0.5
    result = n_for_mean_length(u_crit, s_star, l_max)
    assert isinstance(result, dict)
