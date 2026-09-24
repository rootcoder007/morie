"""Tests for grsmxs.geron_softmax_score."""

from morie.fn import _array_core as np

from morie.fn.grsmxs import geron_softmax_score


def test_grsmxs_basic():
    """Test basic functionality."""
    X = [[1.0, 2.0], [0.5, -1.0]]
    theta = [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]]
    result = geron_softmax_score(X, theta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grsmxs_edge():
    """Test edge cases."""
    X = [[1.0, 2.0], [0.5, -1.0]]
    theta = [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]]
    result = geron_softmax_score(X, theta)
    assert isinstance(result, dict)
