"""Tests for gblup_gebv.gblup_gebv."""

from morie.fn import _array_core as np

from morie.fn.gblup_gebv import gblup_gebv


def _symmetrize(M, n):
    """Average M with its transpose to make it symmetric."""
    return [[(M[i][j] + M[j][i]) / 2 for j in range(n)] for i in range(n)]


def test_msm242_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    Z = rng.normal(0, 1, (n, n))
    G = _symmetrize(Z, n)
    sigma2_g = 0.5
    result = gblup_gebv(X, y, G, sigma2_g)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "gebv" in result
    assert "beta" in result


def test_msm242_edge():
    """Test edge cases."""
    rng = np.random.default_rng(7)
    n, p = 12, 2
    X = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    Z = rng.normal(0, 1, (n, n))
    G = _symmetrize(Z, n)
    sigma2_g = 0.8
    result = gblup_gebv(X, y, G, sigma2_g, sigma2_e=2.0)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "gebv" in result
