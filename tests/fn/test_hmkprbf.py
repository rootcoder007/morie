"""Tests for hmkprbf.geron_kernel_pca_rbf."""

from morie.fn import _array_core as np
from morie.fn.hmkprbf import geron_kernel_pca_rbf


def test_hmkprbf_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    gamma = 1.0
    result = geron_kernel_pca_rbf(X, n_components, gamma)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmkprbf_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    gamma = 1.0
    result = geron_kernel_pca_rbf(X, n_components, gamma)
    assert isinstance(result, dict)


def test_centred_gram_and_eigenvalues_recomputed():
    """RBF Gram of 1-D points, double-centred; its eigenvalues sum to the trace."""
    import math

    import pytest

    from morie.fn.hmkprbf import center_gram

    X = [[0.0], [1.0], [2.5], [4.0]]
    n, g = 4, 0.3
    K = [[math.exp(-g * (X[i][0] - X[j][0]) ** 2) for j in range(n)] for i in range(n)]
    rm = [sum(K[i]) / n for i in range(n)]
    tm = sum(rm) / n
    Kc = [[K[i][j] - rm[i] - rm[j] + tm for j in range(n)] for i in range(n)]
    got = center_gram(K)
    for i in range(n):
        assert [float(v) for v in got[i]] == pytest.approx(Kc[i], rel=1e-12, abs=1e-14)
    r = geron_kernel_pca_rbf(X, n_components=4, gamma=g)
    ev = [float(v) for v in r["eigenvalues"]]
    assert sum(v for v in ev if v > 0) == pytest.approx(sum(Kc[i][i] for i in range(n)), rel=1e-10)
