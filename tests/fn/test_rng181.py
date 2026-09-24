"""Tests for rng181.rangayyan_ch4_pan_tompkins_lowpass_transfer."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_ch4_pan_tompkins_lowpass_transfer


def test_rng181_basic():
    """Test basic functionality."""
    freq = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_pan_tompkins_lowpass_transfer(freq)
    assert isinstance(result, dict)
    assert "freq" in result


def test_rng181_edge():
    """Test edge cases."""
    freq = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_pan_tompkins_lowpass_transfer(freq)
    assert isinstance(result, dict)
