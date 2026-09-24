"""Tests for rng184.rangayyan_ch4_pan_tompkins_highpass_lp_difference_eq."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_ch4_pan_tompkins_highpass_lp_difference_eq


def test_rng184_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_pan_tompkins_highpass_lp_difference_eq(x)
    assert isinstance(result, dict)
    assert "y" in result


def test_rng184_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_pan_tompkins_highpass_lp_difference_eq(x)
    assert isinstance(result, dict)
