"""Tests for volharm.vol_harmonic_volatility."""

from morie.fn import _array_core as np

from morie.fn.volharm import vol_harmonic_volatility


def test_volharm_basic():
    """Test basic functionality."""
    sigma = 0.1
    result = vol_harmonic_volatility(sigma)
    assert isinstance(result, dict)
    assert "harmonic" in result


def test_volharm_edge():
    """Test edge cases."""
    sigma = 0.1
    result = vol_harmonic_volatility(sigma)
    assert isinstance(result, dict)
