"""Tests for rgdnot.rangayyan_dicrotic_notch."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_dicrotic_notch


def test_rgdnot_basic():
    """Test basic functionality."""
    cp = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_dicrotic_notch(cp, fs)
    assert isinstance(result, dict)
    assert "notch" in result


def test_rgdnot_edge():
    """Test edge cases."""
    cp = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_dicrotic_notch(cp, fs)
    assert isinstance(result, dict)
