"""Tests for grlrnc.geron_learning_curves."""

from morie.fn import _array_core as np

from morie.fn.grlrnc import geron_learning_curves


def test_grlrnc_basic():
    """Test basic functionality."""
    X = [[1.0, float(i)] for i in range(12)]
    y = [2.0 * i + 1 for i in range(12)]
    result = geron_learning_curves(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grlrnc_edge():
    """Test edge cases."""
    X = [[1.0, float(i)] for i in range(12)]
    y = [2.0 * i + 1 for i in range(12)]
    result = geron_learning_curves(X, y)
    assert isinstance(result, dict)
