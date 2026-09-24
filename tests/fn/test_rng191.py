"""Tests for rng191.rangayyan_ch4_pan_tompkins_thresholds."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_ch4_pan_tompkins_thresholds


def test_rng191_basic():
    """Test basic functionality."""
    peaki = 0.5
    spki = 0.5
    npki = 0.5
    issignal = 0.5
    result = rangayyan_ch4_pan_tompkins_thresholds(peaki, spki, npki, issignal)
    assert isinstance(result, dict)
    assert "spki" in result


def test_rng191_edge():
    """Test edge cases."""
    peaki = 0.5
    spki = 0.5
    npki = 0.5
    issignal = 0.5
    result = rangayyan_ch4_pan_tompkins_thresholds(peaki, spki, npki, issignal)
    assert isinstance(result, dict)
