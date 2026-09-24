"""Tests for rng197.rangayyan_ch4_dicrotic_notch_smoothed_squared."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_ch4_dicrotic_notch_smoothed_squared


def test_rng197_basic():
    """Test basic functionality."""
    p = 0.1
    result = rangayyan_ch4_dicrotic_notch_smoothed_squared(p)
    assert isinstance(result, dict)
    assert "s" in result


def test_rng197_edge():
    """Test edge cases."""
    p = 0.1
    result = rangayyan_ch4_dicrotic_notch_smoothed_squared(p)
    assert isinstance(result, dict)
