"""Tests for appnp.appnp."""

from morie.fn import _array_core as np

from morie.fn.appnp import appnp


def test_appnp_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, c = 10, 3
    # Build a symmetric positive adjacency matrix (symmetrised internally)
    A_raw = rng.uniform(0, 1, (n, n))
    A = [[(A_raw[i][j] + A_raw[j][i]) / 2.0 for j in range(n)] for i in range(n)]
    # Per-node predictions f_theta(X)
    H = rng.normal(0, 1, (n, c))
    result = appnp(A, H, alpha=0.1, K=10)
    assert isinstance(result, dict)
    assert "Z" in result
    assert "alpha" in result
    assert "K" in result
    assert "n" in result
    assert "c" in result
    Z = result["Z"]
    assert len(Z) == n
    assert len(Z[0]) == c


def test_appnp_edge():
    """Test edge cases using exact closed-form PPNP."""
    rng = np.random.default_rng(42)
    n, c = 5, 2
    A_raw = rng.uniform(0, 1, (n, n))
    A = [[(A_raw[i][j] + A_raw[j][i]) / 2.0 for j in range(n)] for i in range(n)]
    H = rng.normal(0, 1, (n, c))
    result = appnp(A, H, alpha=0.5, K=3, exact=True, softmax=False)
    assert isinstance(result, dict)
    assert "Z" in result
    assert result["exact"] is True
    Z = result["Z"]
    assert len(Z) == n
    assert len(Z[0]) == c
