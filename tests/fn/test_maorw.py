"""Tests for maorw.ma_orwin_fsn."""

from morie.fn import _array_core as np

from morie.fn.maorw import ma_orwin_fsn


def test_maorw_basic():
    """Test basic functionality."""
    d_obs = 0.5
    d_crit = 0.5
    d_filldraw = 5
    k = 5
    result = ma_orwin_fsn(d_obs, d_crit, d_filldraw, k)
    assert isinstance(result, dict)
    assert "Nfs" in result


def test_maorw_edge():
    """Test edge cases."""
    d_obs = 0.5
    d_crit = 0.5
    d_filldraw = 5
    k = 5
    result = ma_orwin_fsn(d_obs, d_crit, d_filldraw, k)
    assert isinstance(result, dict)
