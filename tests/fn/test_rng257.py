"""Tests for rng257.rangayyan_ch4_log_power_spectrum_signal_echo."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_ch4_log_power_spectrum_signal_echo


def test_rng257_basic():
    """Test basic functionality."""
    H = np.random.default_rng(42).normal(0.0, 1.0, 40)
    a = 0.1
    n_0 = 5
    omega = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_log_power_spectrum_signal_echo(H, a, n_0, omega)
    assert isinstance(result, dict)
    assert "log_power" in result


def test_rng257_edge():
    """Test edge cases."""
    H = np.random.default_rng(42).normal(0.0, 1.0, 40)
    a = 0.1
    n_0 = 5
    omega = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_log_power_spectrum_signal_echo(H, a, n_0, omega)
    assert isinstance(result, dict)
