"""Tests for comemb.node2vec."""

from morie.fn import _array_core as np

from morie.fn.comemb import node2vec


def _ring_adjacency(n):
    """Build an n x n adjacency matrix for a cycle graph (1-D ring)."""
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        A[i][(i + 1) % n] = 1.0
        A[i][(i - 1) % n] = 1.0
    return A


def test_comemb_basic():
    """Test basic functionality on a small cycle graph."""
    G = _ring_adjacency(6)
    p = 1.0
    q = 1.0
    dim = 4
    result = node2vec(G, p, q, dim)

    # Documented return type and keys.
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "embedding" in result
    assert "walks" in result
    assert "degree" in result
    assert "n" in result
    assert "dim" in result
    assert "p" in result
    assert "q" in result

    # Documented scalar / structural fields.
    assert result["n"] == 6
    assert result["dim"] == dim
    assert result["p"] == p
    assert result["q"] == q
    assert len(result["degree"]) == 6
    # Ring graph: every node has exactly two neighbours.
    assert all(d == 2 for d in result["degree"])

    # Embedding has shape (n, dim).
    emb = result["embedding"]
    assert len(emb) == 6
    for row in emb:
        assert len(row) == dim


def test_comemb_edge():
    """Edge case: p = q = 1 must reproduce the un-biased (DeepWalk) regime
    and must not raise on a small ring."""
    G = _ring_adjacency(5)
    p = 1.0
    q = 1.0
    dim = 3
    result = node2vec(G, p, q, dim)

    assert isinstance(result, dict)
    assert result["n"] == 5
    assert result["dim"] == dim
    assert result["p"] == p
    assert result["q"] == q

    # Cosine estimate is finite on a connected toy graph.
    est = result["estimate"]
    assert isinstance(est, float)
    import math
    assert not math.isnan(est)

    # Number of walks equals n_walks * n_nodes with the defaults
    # (n_walks=4, walk_len=10) declared in the docstring.
    assert len(result["walks"]) == 4 * 5
