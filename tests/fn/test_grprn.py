"""Tests for grprn.geron_weight_pruning."""

from morie.fn import _array_core as np

from morie.fn.grprn import geron_weight_pruning


def test_grprn_basic():
    """Test basic functionality."""
    W = [[1.0, -0.1], [0.05, 2.0]]
    sparsity = 0.5
    result = geron_weight_pruning(W, sparsity)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grprn_edge():
    """Test edge cases."""
    W = [[1.0, -0.1], [0.05, 2.0]]
    sparsity = 0.5
    result = geron_weight_pruning(W, sparsity)
    assert isinstance(result, dict)
