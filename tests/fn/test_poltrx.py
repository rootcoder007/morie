"""Tests for poltrx.polya_tree_extended."""

from morie.fn import _array_core as np

from morie.fn.poltrx import polya_tree_extended


def test_poltrx_basic():
    """Test basic functionality."""
    levels = 5
    result = polya_tree_extended(levels)
    assert isinstance(result, dict)
    assert "Y" in result


def test_poltrx_edge():
    """Test edge cases."""
    levels = 5
    result = polya_tree_extended(levels)
    assert isinstance(result, dict)
