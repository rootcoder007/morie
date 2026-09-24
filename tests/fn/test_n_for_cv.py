"""Tests for n_for_cv.n_for_cv."""

from morie.fn import _array_core as np

from morie.fn.n_for_cv import (
    n_for_cv,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e10_basic():
    """Test basic functionality."""
    u_crit = 0.5
    cv_star = 0.5
    r_max = 0.5
    result = n_for_cv(u_crit, cv_star, r_max)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e10_edge():
    """Test edge cases."""
    u_crit = 0.5
    cv_star = 0.5
    r_max = 0.5
    result = n_for_cv(u_crit, cv_star, r_max)
    assert isinstance(result, dict)
