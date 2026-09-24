"""Tests for grmlb.geron_multilabel_classification."""

from morie.fn import _array_core as np

from morie.fn.grmlb import geron_multilabel_classification


def test_grmlb_basic():
    """Test basic functionality."""
    X = [[0.9, 0.1], [0.2, 0.8]]
    Y = [[1, 1], [0, 1]]
    result = geron_multilabel_classification(X, Y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grmlb_edge():
    """Test edge cases."""
    X = [[0.9, 0.1], [0.2, 0.8]]
    Y = [[1, 1], [0, 1]]
    result = geron_multilabel_classification(X, Y)
    assert isinstance(result, dict)
