"""Tests for rghmm.rangayyan_fitzhugh_nagumo."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_fitzhugh_nagumo


def test_rghmm_basic():
    """Test basic functionality."""
    result = rangayyan_fitzhugh_nagumo()
    assert isinstance(result, dict)
    assert "t" in result


def test_rghmm_edge():
    """Test edge cases."""
    result = rangayyan_fitzhugh_nagumo()
    assert isinstance(result, dict)
