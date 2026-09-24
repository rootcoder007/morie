"""Tests for rejct.rejection_point."""

from morie.fn import _array_core as np

from morie.fn.rejct import rejection_point


def test_rejct_basic():
    """Test basic functionality."""
    result = rejection_point()
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_rejct_edge():
    """Test edge cases."""
    result = rejection_point()
    assert isinstance(result, dict)
