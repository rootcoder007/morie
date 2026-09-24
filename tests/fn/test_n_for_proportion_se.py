"""Tests for n_for_proportion_se.n_for_proportion_se."""

from morie.fn import _array_core as np

from morie.fn.n_for_proportion_se import (
    n_for_proportion_se,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e3_basic():
    """Test basic functionality."""
    p_star = 0.5
    se_max = 0.5
    result = n_for_proportion_se(p_star, se_max)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e3_edge():
    """Test edge cases."""
    p_star = 0.5
    se_max = 0.5
    result = n_for_proportion_se(p_star, se_max)
    assert isinstance(result, dict)
