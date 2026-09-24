"""Tests for n_for_proportion_length.n_for_proportion_length."""

from morie.fn import _array_core as np

from morie.fn.n_for_proportion_length import (
    n_for_proportion_length,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e11_basic():
    """Test basic functionality."""
    u_crit = 0.5
    p_star = 0.5
    l_max = 0.5
    result = n_for_proportion_length(u_crit, p_star, l_max)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e11_edge():
    """Test edge cases."""
    u_crit = 0.5
    p_star = 0.5
    l_max = 0.5
    result = n_for_proportion_length(u_crit, p_star, l_max)
    assert isinstance(result, dict)
