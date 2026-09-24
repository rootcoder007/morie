"""Tests for kgnn.r_gcn."""

from morie.fn import _array_core as np

from morie.fn.kgnn import r_gcn


def test_kgnn_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p, q = 40, 3, 4
    X = rng.normal(0, 1, (n, p))
    A_r = [rng.normal(0, 1, (n, n)), rng.normal(0, 1, (n, n))]
    W_r = [rng.normal(0, 1, (p, q)), rng.normal(0, 1, (p, q))]
    result = r_gcn(A_r, X, W_r)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["n"] == n
    assert result["relations"] == 2
    assert len(result["H"]) == n
    assert len(result["H"][0]) == q


def test_kgnn_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p, q = 5, 2, 3
    X = rng.normal(0, 1, (n, p))
    A_r = [rng.normal(0, 1, (n, n))]
    W_r = [rng.normal(0, 1, (p, q))]
    W0 = rng.normal(0, 1, (p, q))
    result = r_gcn(A_r, X, W_r, W0=W0)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["n"] == n
    assert result["relations"] == 1
    assert len(result["H"]) == n
    assert len(result["H"][0]) == q
