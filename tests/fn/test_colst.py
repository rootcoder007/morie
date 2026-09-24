"""Tests for colst.collider."""

from morie.fn import _array_core as np

from morie.fn.colst import collider


def test_colst_basic():
    """Test basic functionality."""
    adj = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ])
    # Convert adjacency matrix to edgelist of (parent, child) pairs
    n = len(adj)
    dag = [(u, v) for u in range(n) for v in range(n) if adj[u][v]]
    triple = [0, 2, 1]
    result = collider(dag, triple)
    assert isinstance(result, dict)


def test_colst_edge():
    """Test edge cases."""
    # Simple chain: 0 -> 1 -> 2 (not a collider)
    dag = [(0, 1), (1, 2)]
    triple = [0, 1, 2]
    result = collider(dag, triple)
    assert isinstance(result, dict)
