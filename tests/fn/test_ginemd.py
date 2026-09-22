"""Tests for ginemd.graph_isomorphism_net."""

from morie.fn import _array_core as np

from morie.fn.ginemd import graph_isomorphism_net


def test_ginemd_basic():
    """Test basic functionality."""
    n = 10
    dim = 5
    G = np.eye(n)
    X = np.random.default_rng(42).normal(0, 1, (n, dim))
    eps = 0.0
    result = graph_isomorphism_net(G, X, eps)
    assert isinstance(result, dict)
    assert "H" in result
    assert "eps" in result
    assert result["n_nodes"] == n
    assert result["dim"] == dim

    # GIN aggregation: H_out[v] = (1 + eps) * H[v] + sum_u A[v,u] * H[u]
    # With A = I (identity) and eps = 0: H_out[v] = H[v] + H[v] = 2 * H[v]
    expected = 2.0 * X
    np.testing.assert_allclose(result["H"], expected)


def test_ginemd_edge():
    """Test edge cases."""
    n = 10
    dim = 5
    G = np.eye(n)
    X = np.random.default_rng(42).normal(0, 1, (n, dim))
    eps = 0.5
    result = graph_isomorphism_net(G, X, eps)
    assert isinstance(result, dict)
    assert "H" in result
    assert "eps" in result
    assert result["eps"] == eps
    assert result["n_nodes"] == n
    assert result["dim"] == dim

    # GIN aggregation: H_out[v] = (1 + eps) * H[v] + sum_u A[v,u] * H[u]
    # With A = I and eps = 0.5: H_out[v] = 1.5 * H[v] + H[v] = 2.5 * H[v]
    expected = (1.0 + eps + 1.0) * X
    np.testing.assert_allclose(result["H"], expected)
