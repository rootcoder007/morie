"""Tests for deepwk.deepwalk."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.deepwk import deepwalk


def test_deepwk_basic():
    """Test basic functionality."""
    # 10-node cycle graph: each node connected to neighbours 1 and -1
    n = 10
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        A[i][(i - 1) % n] = 1.0
        A[i][(i + 1) % n] = 1.0
    G = A

    walk_len = 10
    dim = 8
    result = deepwalk(G, walk_len, dim)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result

    # Check documented keys are present
    assert "embedding" in result
    assert "walks" in result
    assert "n_walks_total" in result
    assert "degree" in result
    assert "n" in result
    assert "dim" in result

    # n_walks_total should equal n * n_walks (default n_walks=4)
    n_walks_default = 4
    assert result["n_walks_total"] == n * n_walks_default
    assert result["n"] == n
    assert result["dim"] == dim

    # degree: each node in the cycle has degree 2
    expected_degrees = [2] * n
    assert result["degree"] == expected_degrees

    # Each walk should have walk_len nodes
    assert len(result["walks"]) == n * n_walks_default
    for w in result["walks"]:
        assert len(w) == walk_len
        for v in w:
            assert 0 <= v < n

    # estimate is mean cosine similarity between connected node embeddings
    assert isinstance(result["estimate"], float)


def test_deepwk_edge():
    """Test edge cases."""
    # Small path graph: 0 - 1 - 2 - 3
    n = 4
    A = [[0.0] * n for _ in range(n)]
    edges = [(0, 1), (1, 2), (2, 3)]
    for i, j in edges:
        A[i][j] = 1.0
        A[j][i] = 1.0

    walk_len = 5
    dim = 4
    result = deepwalk(A, walk_len, dim, n_walks=2, window=2,
                      epochs=1, lr=0.05, neg=1, seed=7)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "embedding" in result
    assert "n_walks_total" in result
    assert result["n"] == n
    assert result["dim"] == dim
    assert result["n_walks_total"] == n * 2
    assert result["degree"] == [1, 2, 2, 1]
