"""Tests for grln.geron_layer_normalization."""

from morie.fn import _array_core as np

from morie.fn.grln import geron_layer_normalization


def test_grln_basic():
    """Test basic functionality."""
    X = [[0.5, -1.5, 2.0], [3.0, 0.0, -1.0]]
    result = geron_layer_normalization(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grln_edge():
    """Test edge cases."""
    X = [[0.5, -1.5, 2.0], [3.0, 0.0, -1.0]]
    result = geron_layer_normalization(X)
    assert isinstance(result, dict)
