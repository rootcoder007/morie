"""Tests for ergmod.ergm."""

from morie.fn import _array_core as np

from morie.fn.ergmod import ergm


def _make_simple_graph():
    """Build a small 0/1 symmetric adjacency matrix with zero diagonal."""
    # Triangle on nodes 0,1,2 plus edge (2,3)
    # rows: 0..3, symmetric, zero diagonal
    A = [
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ]
    return A


def test_ergmod_basic():
    """Test basic functionality with edges statistic."""
    G = _make_simple_graph()
    statistics = ("edges",)
    theta_init = [0.0]
    result = ergm(G, statistics, theta_init)
    assert isinstance(result, dict)
    # documented return keys
    for key in ("estimate", "theta", "se", "observed_stats",
               "pseudo_loglik", "n_dyads", "iters_used", "n", "method"):
        assert key in result

    # n must match the input size
    assert result["n"] == 4
    # 4 nodes -> C(4,2) = 6 dyads
    assert result["n_dyads"] == 6

    # observed_stats for "edges" on this graph: edges (0,1),(0,2),(1,2),(2,3) -> 4
    assert result["observed_stats"] == [4.0]

    # theta must have one entry per statistic
    assert len(result["theta"]) == 1
    assert len(result["se"]) == 1

    # Independent expectation: Newton-Raphson starting from theta=[0],
    # the first step gradient/Hessian are computable, so iters_used >= 1
    assert result["iters_used"] >= 1


def test_ergmod_edge():
    """Test edge case: default arguments (theta_init=None) on a valid graph."""
    G = _make_simple_graph()
    statistics = ("edges",)
    # Use defaults by passing only G and statistics
    result = ergm(G, statistics)
    assert isinstance(result, dict)

    # Default iters/tol: should converge or hit iters bound without raising
    assert result["n"] == 4
    assert result["n_dyads"] == 6
    assert result["observed_stats"] == [4.0]
    assert result["method"].startswith("Exponential random graph model")
