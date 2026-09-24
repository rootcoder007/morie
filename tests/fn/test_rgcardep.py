"""Tests for rgcardep.rangayyan_cardiac_elecphys."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_cardiac_elecphys


def test_rgcardep_basic():
    """Test basic functionality."""
    result = rangayyan_cardiac_elecphys()
    assert isinstance(result, dict)
    assert "x_cm" in result


def test_rgcardep_edge():
    """Test edge cases."""
    result = rangayyan_cardiac_elecphys()
    assert isinstance(result, dict)
