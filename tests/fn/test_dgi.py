"""Tests for dgi.dgi."""

import pytest

from morie.fn import _array_core as np

from morie.fn.dgi import dgi


def test_dgi_basic():
    """Test basic functionality."""
    n = 10
    f = 5
    d = 4
    rng = np.random.default_rng(42)
    # Build a symmetric 0/1 adjacency matrix
    G = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            val = int(rng.integers(0, 2))
            G[i][j] = val
            G[j][i] = val
    X = rng.normal(0, 1, (n, f))
    encoder = rng.normal(0, 1, (f, d))
    result = dgi(G, X, encoder)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "n" in result
    assert "d" in result


def test_dgi_edge():
    """Test edge cases."""
    with pytest.raises(ValueError):
        dgi([], [])
