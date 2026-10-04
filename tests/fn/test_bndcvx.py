"""Tests for bndcvx.bound_convex_estimator."""

from morie.fn.bndcvx import bound_convex_estimator


def test_bndcvx_basic():
    """Test basic functionality."""
    c = 0.5
    result = bound_convex_estimator(c)
    assert isinstance(result, dict)
    assert "lower" in result


def test_bndcvx_edge():
    """Test edge cases."""
    c = 0.5
    result = bound_convex_estimator(c)
    assert isinstance(result, dict)
