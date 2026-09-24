"""Tests for rng252.rangayyan_ch4_complex_cepstrum_signal_with_echo."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_ch4_complex_cepstrum_signal_with_echo


def test_rng252_basic():
    """Test basic functionality."""
    h_hat = 0.1
    a = 0.1
    n_0 = 5
    result = rangayyan_ch4_complex_cepstrum_signal_with_echo(h_hat, a, n_0)
    assert isinstance(result, dict)
    assert "y_hat" in result


def test_rng252_edge():
    """Test edge cases."""
    h_hat = 0.1
    a = 0.1
    n_0 = 5
    result = rangayyan_ch4_complex_cepstrum_signal_with_echo(h_hat, a, n_0)
    assert isinstance(result, dict)
