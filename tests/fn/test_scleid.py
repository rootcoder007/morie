"""Tests for scleid.leiden_clustering."""

from morie.fn import _array_core as np

from morie.fn.scleid import leiden_clustering


def test_scleid_basic():
    """Test basic functionality."""
    graph = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = leiden_clustering(graph)
    assert isinstance(result, dict)
    assert "estimate" in result or "labels" in result


def test_scleid_edge():
    """Test edge cases."""
    graph = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = leiden_clustering(graph)
    assert isinstance(result, dict)
