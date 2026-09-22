"""Tests for densty.density."""

from morie.fn import _array_core as np

from morie.fn.densty import density


def test_densty_basic():
    """Test basic functionality on a triangle (K3)."""
    # K3: 3 vertices, 3 edges, density = 3 / C(3,2) = 3 / 3 = 1.0
    K3 = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    G = np.asarray(K3, dtype=float)
    result = density(G)
    assert isinstance(result, dict)
    assert "estimate" in result
    n = G.shape[0]
    edges = int(np.sum(G) // 2)
    possible = n * (n - 1) // 2
    expected_estimate = float(edges / possible)
    assert result["estimate"] == expected_estimate
    assert result["n_edges"] == edges
    assert result["n_possible"] == int(possible)
    assert result["n"] == int(n)
    assert "method" in result


def test_densty_edge():
    """Test edge cases: path on 3 vertices (2 of 3 possible edges)."""
    # Path P3: density = 2 / C(3,2) = 2 / 3
    path = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    G = np.asarray(path, dtype=float)
    result = density(G)
    assert isinstance(result, dict)
    n = G.shape[0]
    edges = int(np.sum(G) // 2)
    possible = n * (n - 1) // 2
    expected_estimate = float(edges / possible)
    assert result["estimate"] == expected_estimate
    assert result["n_edges"] == edges
    assert result["n_possible"] == int(possible)
