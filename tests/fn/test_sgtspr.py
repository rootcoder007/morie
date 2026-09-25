"""Tests for sgtspr.sgt_spectral_radius_bound."""

from morie.fn.sgtspr import sgt_spectral_radius_bound


def test_sgtspr_basic():
    """A path is bipartite, so every odd trace(A^k) vanishes."""
    A = [[0.0, 1.0, 0.0],
         [1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0]]
    r = sgt_spectral_radius_bound(A)
    assert r["bipartite"] is True
    assert [float(v) for v in r["evidence"]] == [0.0, 0.0]
    assert float(r["max_odd_trace"]) == 0.0
    assert sorted(int(s) for s in r["part_sizes"]) == [1, 2]
    assert r["n"] == 3
    assert float(r["m"]) == 2.0
    assert r["n_components"] == 1
    # the colouring is a valid 2-colouring: adjacent nodes differ in sign
    col = [int(c) for c in r["colouring"]]
    for i in range(3):
        for j in range(3):
            if A[i][j]:
                assert col[i] == -col[j]


def test_sgtspr_triangle():
    """A triangle has trace(A^3) = 6, so it is not bipartite."""
    A = [[0.0, 1.0, 1.0],
         [1.0, 0.0, 1.0],
         [1.0, 1.0, 0.0]]
    r = sgt_spectral_radius_bound(A)
    assert r["bipartite"] is False
    ev = [float(v) for v in r["evidence"]]
    assert abs(ev[0]) < 1e-9
    assert abs(ev[1] - 6.0) < 1e-9
    assert abs(float(r["max_odd_trace"]) - 6.0) < 1e-9


def test_sgtspr_edge():
    """A 4-cycle is bipartite with parts of size 2 and 2."""
    A = [[0.0, 1.0, 0.0, 1.0],
         [1.0, 0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 1.0],
         [1.0, 0.0, 1.0, 0.0]]
    r = sgt_spectral_radius_bound(A)
    assert r["bipartite"] is True
    assert sorted(int(s) for s in r["part_sizes"]) == [2, 2]
    assert float(r["m"]) == 4.0
