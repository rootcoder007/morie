"""Tests for scleid.leiden_clustering."""

from morie.fn.scleid import leiden_clustering


def test_scleid_basic():
    """Test basic functionality."""
    graph = [
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ]
    result = leiden_clustering(graph)
    assert isinstance(result, dict)
    assert result["labels"] == [0, 0, 0, 1, 1, 1]
    assert result["connected"] and result["n_communities"] == 2
    assert abs(result["estimate"] - 5 / 14) < 1e-12  # two triangles joined by one edge: 2(6/14) - 2(7/14)^2


def test_scleid_edge():
    """Test edge cases."""
    graph = [
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ]
    result = leiden_clustering(graph)
    assert isinstance(result, dict)
