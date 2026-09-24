"""Tests for hmxln.geron_xlnet."""

from morie.fn import _array_core as np

from morie.fn.hmxln import geron_xlnet


def test_hmxln_basic():
    """Test basic functionality."""
    X = [0, 1, 2, 1]
    result = geron_xlnet(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmxln_edge():
    """Test edge cases."""
    X = [0, 1, 2, 1]
    result = geron_xlnet(X)
    assert isinstance(result, dict)
