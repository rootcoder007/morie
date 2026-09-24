"""Tests for merck.mercer_theorem."""

import math

from morie.fn import _array_core as np

from morie.fn.merck import mercer_theorem


def test_merck_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 10, 3
    X = rng.normal(0, 1, (n, p))
    # Compute Gram matrix K = X X^T (symmetric positive semi-definite kernel matrix)
    K = [[sum(X[i][k] * X[j][k] for k in range(p)) for j in range(n)] for i in range(n)]
    result = mercer_theorem(K)
    assert "eigenvalues" in result
    eigenvalues = result["eigenvalues"]
    assert isinstance(eigenvalues, list)
    assert len(eigenvalues) == n
    assert all(math.isfinite(float(v)) for v in eigenvalues)


def test_merck_edge():
    """Test edge cases."""
    # 1x1 kernel matrix
    K = [[42.0]]
    result = mercer_theorem(K)
    assert result["n"] == 1
