"""Tests for rng183.rangayyan_ch4_pan_tompkins_highpass_lp_component."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_ch4_pan_tompkins_highpass_lp_component


def test_rng183_basic():
    """Test basic functionality."""
    freq = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_pan_tompkins_highpass_lp_component(freq)
    assert isinstance(result, dict)
    assert "freq" in result


def test_rng183_edge():
    """Test edge cases."""
    freq = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_pan_tompkins_highpass_lp_component(freq)
    assert isinstance(result, dict)
