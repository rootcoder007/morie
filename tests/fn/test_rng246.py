"""Tests for rng246.rangayyan_ch4_signal_with_echo_input."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_ch4_signal_with_echo_input


def test_rng246_basic():
    """Test basic functionality."""
    a = 0.1
    n_0 = 5
    n = 5
    result = rangayyan_ch4_signal_with_echo_input(a, n_0, n)
    assert isinstance(result, dict)
    assert "x" in result


def test_rng246_edge():
    """Test edge cases."""
    a = 0.1
    n_0 = 5
    n = 5
    result = rangayyan_ch4_signal_with_echo_input(a, n_0, n)
    assert isinstance(result, dict)
