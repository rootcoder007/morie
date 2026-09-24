"""Tests for gcnchb.chebnet."""

from morie.fn import _array_core as np

from morie.fn.gcnchb import chebnet


def test_gcnchb_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 10, 3
    B = rng.normal(0, 1, (n, n))
    L = [[B[i][j] + B[j][i] for j in range(n)] for i in range(n)]
    for i in range(n):
        L[i][i] += n + 1.0
    X = rng.normal(0, 1, (n, p))
    result = chebnet(L, X, K=3)
    assert isinstance(result, dict)
    assert "estimate" in result
    H = result["H"]
    assert len(H) == n
    assert len(H[0]) == p
    assert result["K"] == 3


def test_gcnchb_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 6, 2
    B = rng.normal(0, 1, (n, n))
    L = [[B[i][j] + B[j][i] for j in range(n)] for i in range(n)]
    for i in range(n):
        L[i][i] += n + 1.0
    X = rng.normal(0, 1, (n, p))
    result = chebnet(L, X, K=1)
    assert isinstance(result, dict)
    assert "lambda_max" in result
    assert result["K"] == 1
    assert result["n"] == n
    assert len(result["H"]) == n
