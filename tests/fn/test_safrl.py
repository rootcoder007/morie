"""Tests for safrl.safe_rl."""

from morie.fn import _array_core as np

from morie.fn.safrl import safe_rl


def test_safrl_basic():
    """Test basic functionality."""
    g = 0.5
    H = 0.5
    result = safe_rl(g, H)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_safrl_edge():
    """Test edge cases."""
    g = 0.5
    H = 0.5
    result = safe_rl(g, H)
    assert isinstance(result, dict)
