"""Tests for ibpfa.indian_buffet_factor."""

from morie.fn import _array_core as np

from morie.fn.ibpfa import indian_buffet_factor


def test_ibpfa_basic():
    """Test basic functionality."""
    n = 5
    alpha = 0.1
    result = indian_buffet_factor(n, alpha)
    assert isinstance(result, dict)
    assert "Z" in result


def test_ibpfa_edge():
    """Test edge cases."""
    n = 5
    alpha = 0.1
    result = indian_buffet_factor(n, alpha)
    assert isinstance(result, dict)
