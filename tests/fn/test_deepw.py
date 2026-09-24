"""Tests for deepw.deepwalk."""

import math

from morie.fn.deepw import deepwalk


def _ring_graph(n):
    """Build an n-node ring graph adjacency matrix."""
    G = [[0] * n for _ in range(n)]
    for i in range(n):
        G[i][(i + 1) % n] = 1
        G[i][(i - 1) % n] = 1
    return G


def test_deepw_basic():
    """Test basic functionality."""
    G = _ring_graph(10)
    result = deepwalk(G, walk_len=6, dim=4, n_walks=2, window=2, epochs=1, lr=0.05, neg=2, seed=42)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "embedding" in result
    assert "walks" in result
    assert "n_walks_total" in result
    assert "degree" in result
    assert "n" in result
    assert "dim" in result
    assert "method" in result
    assert result["n"] == 10
    assert result["dim"] == 4
    assert result["n_walks_total"] == 20
    assert len(result["degree"]) == 10
    assert all(d == 2 for d in result["degree"])
    assert math.isfinite(float(result["estimate"]))


def test_deepw_edge():
    """Test edge cases."""
    G = _ring_graph(5)
    result = deepwalk(G, walk_len=2, dim=1, n_walks=1, window=1, epochs=1, lr=0.1, neg=1, seed=0)
    assert isinstance(result, dict)
    assert result["n"] == 5
    assert result["dim"] == 1
    assert result["n_walks_total"] == 5
    assert len(result["degree"]) == 5
