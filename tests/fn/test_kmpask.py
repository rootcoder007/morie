"""Tests for kmpask.kamath_pass_at_k."""

from morie.fn import _array_core as np

from morie.fn.kmpask import kamath_pass_at_k


def test_kmpask_basic():
    """Test basic functionality."""
    n = 5
    c = 0.5
    k = 5
    result = kamath_pass_at_k(n, c, k)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_kmpask_edge():
    """Test edge cases."""
    n = 5
    c = 0.5
    k = 5
    result = kamath_pass_at_k(n, c, k)
    assert isinstance(result, dict)
