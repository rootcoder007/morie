"""Tests for rng038.rangayyan_ch3_test_signal_sin_cos."""

from morie.fn import _array_core as np

from morie.fn.bsasig import rangayyan_ch3_test_signal_sin_cos


def test_rng038_basic():
    """Test basic functionality."""
    result = rangayyan_ch3_test_signal_sin_cos()
    assert isinstance(result, dict)
    assert "x" in result or "x" in result


def test_rng038_edge():
    """Test edge cases."""
    result = rangayyan_ch3_test_signal_sin_cos()
    assert isinstance(result, dict)
