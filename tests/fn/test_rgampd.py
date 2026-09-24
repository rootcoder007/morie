"""Tests for rgampd.rangayyan_amplitude_demod."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_amplitude_demod


def test_rgampd_basic():
    """Test basic functionality."""
    x = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = rangayyan_amplitude_demod(x)
    assert isinstance(result, dict)
    assert "amplitude" in result


def test_rgampd_edge():
    """Test edge cases."""
    x = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = rangayyan_amplitude_demod(x)
    assert isinstance(result, dict)
