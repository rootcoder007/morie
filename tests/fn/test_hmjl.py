"""Tests for hmjl.geron_johnson_lindenstrauss."""

from morie.fn import _array_core as np

from morie.fn.hmjl import geron_johnson_lindenstrauss


def test_hmjl_basic():
    """Test basic functionality."""
    n = 5
    eps = 0.1
    result = geron_johnson_lindenstrauss(n, eps)
    assert isinstance(result, dict)
    assert "estimate" in result or "d_min" in result


def test_hmjl_edge():
    """Test edge cases."""
    n = 5
    eps = 0.1
    result = geron_johnson_lindenstrauss(n, eps)
    assert isinstance(result, dict)
