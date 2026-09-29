"""Tests for krrFDA.kernel_ridge_regression."""

from morie.fn import _array_core as np
from morie.fn.krrFDA import kernel_ridge_regression


def test_krrFDA_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kernel_ridge_regression(X, y)
    assert isinstance(result, dict)
    assert "x_eval" in result


def test_krrFDA_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kernel_ridge_regression(X, y)
    assert isinstance(result, dict)


def test_alias_solves_the_dual_system():
    """alpha = (K + lambda I)^-1 y with the Gaussian Gram matrix."""
    import math

    import pytest

    x = [0.0, 0.5, 1.0, 1.5, 2.0]
    y = [1.0, 1.4, 0.9, 0.2, -0.3]
    h, lam = 0.7, 0.3
    r = kernel_ridge_regression(x, y, lam=lam, bandwidth=h)
    K = [[math.exp(-0.5 * ((a - b) / h) ** 2) / math.sqrt(2 * math.pi) for b in x] for a in x]
    al = r["alpha"]
    for i in range(5):
        assert sum(K[i][j] * al[j] for j in range(5)) + lam * al[i] == pytest.approx(y[i], rel=1e-10, abs=1e-12)
