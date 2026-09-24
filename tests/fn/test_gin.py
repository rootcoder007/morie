"""Tests for gin.gin."""

from morie.fn import _array_core as np

from morie.fn.gin import gin


def test_gin_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    # Create symmetric 0/1 adjacency matrix
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            val = rng.integers(0, 2)
            A[i][j] = val
            A[j][i] = val
    X = rng.normal(0, 1, (n, p))
    epsilon = 1e-6
    result = gin(A, X, epsilon)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_gin_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 5, 2
    # Create symmetric 0/1 adjacency matrix
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            val = rng.integers(0, 2)
            A[i][j] = val
            A[j][i] = val
    X = rng.normal(0, 1, (n, p))
    epsilon = 1e-6
    result = gin(A, X, epsilon)
    assert isinstance(result, dict)
