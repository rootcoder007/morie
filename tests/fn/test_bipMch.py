"""Tests for bipMch.bipartite_matching."""

from morie.fn import _array_core as np

from morie.fn.bipMch import bipartite_matching


def test_bipMch_basic():
    """Test basic functionality with a perfect matching."""
    edges = [(0, 0), (1, 1), (2, 2)]
    result = bipartite_matching(edges)
    assert isinstance(result, dict)
    assert "size" in result
    assert "is_perfect" in result
    assert "matching" in result
    assert "n_unmatched_left" in result
    assert result["size"] == 3
    assert bool(result["is_perfect"]) is True


def test_bipMch_edge():
    """Test edge case with unmatched left vertices."""
    edges = [(0, 0), (1, 0), (2, 0)]
    result = bipartite_matching(edges)
    assert isinstance(result, dict)
    assert "size" in result
    assert "n_unmatched_left" in result
    assert "matching" in result
    assert "is_perfect" in result
    assert result["size"] == 1
    assert result["n_unmatched_left"] == 2
