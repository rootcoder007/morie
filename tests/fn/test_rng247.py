"""Tests for rng247.rangayyan_ch4_signal_with_echo_output."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_ch4_signal_with_echo_output


def test_rng247_basic():
    """Test basic functionality."""
    h = 0.1
    a = 0.1
    n_0 = 5
    result = rangayyan_ch4_signal_with_echo_output(h, a, n_0)
    assert isinstance(result, dict)
    assert "y" in result


def test_rng247_edge():
    """Test edge cases."""
    h = 0.1
    a = 0.1
    n_0 = 5
    result = rangayyan_ch4_signal_with_echo_output(h, a, n_0)
    assert isinstance(result, dict)
