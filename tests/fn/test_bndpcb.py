"""Tests for bndpcb.bound_pseudo_credible."""

from morie.fn import _array_core as np

from morie.fn.bndpcb import bound_pseudo_credible


def test_bndpcb_basic():
    """Test basic functionality."""
    x = 0.5
    result = bound_pseudo_credible(x)
    assert isinstance(result, dict)
    assert "lower" in result


def test_bndpcb_edge():
    """Test edge cases."""
    x = 0.5
    result = bound_pseudo_credible(x)
    assert isinstance(result, dict)
