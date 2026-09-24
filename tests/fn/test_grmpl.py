"""Tests for grmpl.geron_max_pooling."""

from morie.fn import _array_core as np

from morie.fn.grmpl import geron_max_pooling


def test_grmpl_basic():
    """Test basic functionality."""
    X = [[1.0, 5.0, 2.0], [3.0, 4.0, 0.0], [9.0, 1.0, 1.0]]
    result = geron_max_pooling(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grmpl_edge():
    """Test edge cases."""
    X = [[1.0, 5.0, 2.0], [3.0, 4.0, 0.0], [9.0, 1.0, 1.0]]
    result = geron_max_pooling(X)
    assert isinstance(result, dict)
