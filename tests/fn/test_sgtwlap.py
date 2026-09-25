"""Tests for sgtwlap.sgt_weighted_laplacian."""

from morie.fn.sgtwlap import sgt_weighted_laplacian


def test_sgtwlap_basic():
    """Edge list (u, v, w) with one-based labels assembles W and L = D - W."""
    r = sgt_weighted_laplacian([[1, 2, 1.0], [2, 3, 2.0], [1, 3, 0.5]])
    assert r["n"] == 3
    assert r["m"] == 3
    W = [list(map(float, row)) for row in r["W"]]
    assert W == [[0.0, 1.0, 0.5], [1.0, 0.0, 2.0], [0.5, 2.0, 0.0]]
    assert [float(d) for d in r["degree"]] == [1.5, 3.0, 2.5]
    assert float(r["volume"]) == 7.0
    L = [list(map(float, row)) for row in r["L"]]
    assert L == [[1.5, -1.0, -0.5], [-1.0, 3.0, -2.0], [-0.5, -2.0, 2.5]]
    for row in L:
        assert abs(sum(row)) < 1e-9


def test_sgtwlap_parallel_edges_accumulate():
    """Two rows on the same pair add their weights."""
    r = sgt_weighted_laplacian([[1, 2, 1.0], [1, 2, 2.5]])
    assert float(r["W"][0][1]) == 3.5
    assert [float(d) for d in r["degree"]] == [3.5, 3.5]
    assert float(r["volume"]) == 7.0


def test_sgtwlap_edge():
    """A loop adds to the degree but not to the diagonal: L(v,v) = d_v - w(v,v)."""
    r = sgt_weighted_laplacian([[1, 2, 1.0], [2, 2, 4.0]], n=3)
    assert r["n"] == 3
    assert [float(d) for d in r["degree"]] == [1.0, 5.0, 0.0]
    assert float(r["L"][1][1]) == 1.0
    assert float(r["L"][0][0]) == 1.0
