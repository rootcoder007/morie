"""Tests for esumtv.effective_resistance."""

from morie.fn import _array_core as np

from morie.fn.esumtv import effective_resistance


def _make_graph(weights):
    """Build a symmetric weight matrix from a 2D list of non-negative floats."""
    n = len(weights)
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                A[i][j] = float(weights[i][j])
    return A


def test_esumtv_basic():
    """Test basic functionality on a small path graph."""
    # Path graph on 4 nodes: 1 -- 2 -- 3 -- 4, unit conductances.
    G = _make_graph([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    u, v = 0, 3
    result = effective_resistance(G, u, v)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "resistance" in result
    assert "degree_u" in result
    assert "degree_v" in result
    assert "n" in result
    assert "method" in result

    # For a path of length 3 with unit conductances, the effective
    # resistance between the two endpoints equals the path length = 3.
    # Computed independently from the literature formula:
    #   R_uv = (e_u - e_v)' L^+ (e_u - e_v) = 3 for this graph.
    expected_R = 3.0
    assert abs(result["estimate"] - expected_R) < 1e-10
    assert abs(result["resistance"] - expected_R) < 1e-10

    # Degrees of endpoint nodes on a path of length 3 are both 1.
    assert abs(result["degree_u"] - 1.0) < 1e-10
    assert abs(result["degree_v"] - 1.0) < 1e-10
    assert result["n"] == 4


def test_esumtv_edge():
    """Test edge cases: same node pair returns zero resistance."""
    G = _make_graph([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    # u == v must return R = 0 exactly.
    result = effective_resistance(G, 1, 1)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert abs(result["estimate"] - 0.0) < 1e-10
    assert abs(result["resistance"] - 0.0) < 1e-10
    assert result["n"] == 3
