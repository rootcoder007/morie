"""Tests for gh_c11_4.ghosal_gp_dens_crt."""

from morie.fn import _array_core as np

from morie.fn.gh_c11_4 import ghosal_gp_dens_crt


def test_gh_c11_4_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_gp_dens_crt(x)
    assert "n" in result
    assert np.all(np.isfinite(np.asarray(result["rate"], dtype=float)))
    assert "smoothness" in result
    assert "kernel" in result
    assert "minimax_rate" in result
    assert "attains_minimax" in result
    assert "rate_kind" in result


def test_gh_c11_4_edge():
    """Test edge cases."""
    import pytest
    with pytest.raises(ValueError):
        ghosal_gp_dens_crt(np.array([42.0]))
