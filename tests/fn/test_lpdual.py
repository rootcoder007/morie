"""Tests for lpdual.lp_dual."""

from morie.fn import _array_core as np

from morie.fn.lpdual import lp_dual


def test_lpdual_basic():
    """Test basic functionality."""
    A = 0.5
    b = 0.5
    c = 0.5
    result = lp_dual(A, b, c)
    assert isinstance(result, dict)
    assert "dual_A" in result


def test_lpdual_edge():
    """Test edge cases."""
    A = 0.5
    b = 0.5
    c = 0.5
    result = lp_dual(A, b, c)
    assert isinstance(result, dict)
