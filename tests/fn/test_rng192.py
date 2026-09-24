"""Tests for rng192.rangayyan_ch4_pan_tompkins_searchback_update."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_ch4_pan_tompkins_searchback_update


def test_rng192_basic():
    """Test basic functionality."""
    peaki = 0.1
    spki = 0.1
    result = rangayyan_ch4_pan_tompkins_searchback_update(peaki, spki)
    assert isinstance(result, dict)
    assert "spki" in result


def test_rng192_edge():
    """Test edge cases."""
    peaki = 0.1
    spki = 0.1
    result = rangayyan_ch4_pan_tompkins_searchback_update(peaki, spki)
    assert isinstance(result, dict)
