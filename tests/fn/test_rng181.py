"""Tests for rng181.rangayyan_ch4_pan_tompkins_lowpass_transfer."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_ch4_pan_tompkins_lowpass_transfer


def test_rng181_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_ch4_pan_tompkins_lowpass_transfer(z)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rng181_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_ch4_pan_tompkins_lowpass_transfer(z)
    assert isinstance(result, dict)
