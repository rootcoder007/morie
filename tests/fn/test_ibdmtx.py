"""Tests for ibdmtx.ibd_matrix."""

from morie.fn import _array_core as np

from morie.fn.ibdmtx import ibd_matrix


def test_ibdmtx_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, m = 10, 30
    G = rng.integers(0, 3, (n, m))
    result = ibd_matrix(G)
    assert isinstance(result, dict)
    for key in ("estimate", "Z0", "Z1", "Z2", "n_snps_used", "n", "m", "method"):
        assert key in result
    assert result["n"] == n
    assert result["m"] == m
    est = result["estimate"]
    assert len(est) == n
    for row in est:
        assert len(row) == n
    Z2 = result["Z2"]
    assert len(Z2) == n
    for row in Z2:
        assert len(row) == n


def test_ibdmtx_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, m = 3, 20
    G = rng.integers(0, 3, (n, m))
    result = ibd_matrix(G)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "Z0" in result
    assert "Z1" in result
    assert "Z2" in result
    assert result["n"] == n
    assert result["m"] == m
    est = result["estimate"]
    Z2 = result["Z2"]
    # diagonal entries (an individual with itself) are initialized to 1
    for i in range(n):
        assert est[i][i] == 1.0
        assert Z2[i][i] == 1.0
