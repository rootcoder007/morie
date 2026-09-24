"""Tests for rng250.rangayyan_ch4_log_signal_echo."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_ch4_log_signal_echo


def test_rng250_basic():
    """Test basic functionality."""
    a = 0.1
    n_0 = 5
    omega = 0.1
    result = rangayyan_ch4_log_signal_echo(a, n_0, omega)
    assert isinstance(result, dict)
    assert "Y_hat" in result


def test_rng250_edge():
    """Test edge cases."""
    a = 0.1
    n_0 = 5
    omega = 0.1
    result = rangayyan_ch4_log_signal_echo(a, n_0, omega)
    assert isinstance(result, dict)
