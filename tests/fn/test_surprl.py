"""Tests for surprl.surprisal."""

from morie.fn import _array_core as np

from morie.fn.surprl import surprisal


def test_surprl_basic():
    """Test basic functionality."""
    p = 0.1
    x = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = surprisal(p, x)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_surprl_edge():
    """Test edge cases."""
    p = 0.1
    x = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = surprisal(p, x)
    assert isinstance(result, dict)
