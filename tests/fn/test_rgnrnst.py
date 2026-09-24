"""Tests for rgnrnst.rangayyan_nernst_potential."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_nernst_potential


def test_rgnrnst_basic():
    """Test basic functionality."""
    result = rangayyan_nernst_potential()
    assert isinstance(result, dict)
    assert "ion" in result


def test_rgnrnst_edge():
    """Test edge cases."""
    result = rangayyan_nernst_potential()
    assert isinstance(result, dict)
