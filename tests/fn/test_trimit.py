"""Tests for trimit.weight_trimming."""

from morie.fn import _array_core as np

from morie.fn.trimit import weight_trimming


def test_trimit_basic():
    """Test basic functionality."""
    y = 0.5
    weights = 0.5
    threshold = 0.5
    result = weight_trimming(y, weights, threshold)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_trimit_edge():
    """Test edge cases."""
    y = 0.5
    weights = 0.5
    threshold = 0.5
    result = weight_trimming(y, weights, threshold)
    assert isinstance(result, dict)
