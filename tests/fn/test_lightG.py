"""Tests for lightG.lightgcn."""

from morie.fn import _array_core as np

from morie.fn.lightG import lightgcn


def test_lightG_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    d = 8
    K = 3
    # Symmetric 0/1 adjacency matrix (sparse graph)
    raw = rng.uniform(0, 1, (n, n))
    A = [[1.0 if (raw[i][j] > 0.7 or raw[j][i] > 0.7) else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        A[i][i] = 0.0
    # Initial embeddings: n x d
    E = rng.normal(0, 1, (n, d))
    result = lightgcn(A, E, K)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_lightG_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 20
    d = 4
    K = 1
    # Smaller, sparser graph with single propagation layer
    raw = rng.uniform(0, 1, (n, n))
    A = [[1.0 if (raw[i][j] > 0.85 or raw[j][i] > 0.85) else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        A[i][i] = 0.0
    E = rng.normal(0, 1, (n, d))
    result = lightgcn(A, E, K)
    assert isinstance(result, dict)
    assert len(result) > 0
