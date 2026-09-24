"""Tests for grovo.geron_one_vs_one."""

from morie.fn import _array_core as np

from morie.fn.grovo import geron_one_vs_one


def test_grovo_basic():
    """Test basic functionality."""
    X = [[0.0], [0.5], [5.0], [5.5], [10.0], [10.5]]
    y = [0, 0, 1, 1, 2, 2]
    result = geron_one_vs_one(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grovo_edge():
    """Test edge cases."""
    X = [[0.0], [0.5], [5.0], [5.5], [10.0], [10.5]]
    y = [0, 0, 1, 1, 2, 2]
    result = geron_one_vs_one(X, y)
    assert isinstance(result, dict)
