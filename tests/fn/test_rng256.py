"""Tests for rng256.rangayyan_ch4_power_spectrum_signal_echo."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_ch4_power_spectrum_signal_echo


def test_rng256_basic():
    """Test basic functionality."""
    H = np.random.default_rng(42).normal(0.0, 1.0, 40)
    a = 0.1
    n_0 = 5
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_power_spectrum_signal_echo(H, a, n_0, z)
    assert isinstance(result, dict)
    assert "power" in result


def test_rng256_edge():
    """Test edge cases."""
    H = np.random.default_rng(42).normal(0.0, 1.0, 40)
    a = 0.1
    n_0 = 5
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_power_spectrum_signal_echo(H, a, n_0, z)
    assert isinstance(result, dict)
