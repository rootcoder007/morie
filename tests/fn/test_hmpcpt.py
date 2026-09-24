"""Tests for hmpcpt.geron_perceptron."""

from morie.fn import _array_core as np

from morie.fn.hmpcpt import geron_perceptron


def test_hmpcpt_basic():
    """Test basic functionality."""
    X = 0.5
    y = 1
    result = geron_perceptron(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "w" in result


def test_hmpcpt_edge():
    """Test edge cases."""
    X = 0.5
    y = 1
    result = geron_perceptron(X, y)
    assert isinstance(result, dict)
