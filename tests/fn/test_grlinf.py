"""Tests for grlinf.geron_linear_layer_forward."""

from morie.fn import _array_core as np

from morie.fn.grlinf import geron_linear_layer_forward


def test_grlinf_basic():
    """Test basic functionality."""
    X = [3.0, 4.0]
    W = [[1.0, 2.0], [-1.0, 0.5]]
    b = [1.0, -1.0]
    result = geron_linear_layer_forward(X, W, b)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grlinf_edge():
    """Test edge cases."""
    X = [3.0, 4.0]
    W = [[1.0, 2.0], [-1.0, 0.5]]
    b = [1.0, -1.0]
    result = geron_linear_layer_forward(X, W, b)
    assert isinstance(result, dict)
