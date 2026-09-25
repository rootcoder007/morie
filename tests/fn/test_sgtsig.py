"""Tests for sgtsig.sgt_signless_laplacian."""

from morie.fn.sgtsig import sgt_signless_laplacian


def test_sgtsig_basic():
    """Q = D + A, entry by entry, on the path 0-1-2."""
    A = [[0.0, 1.0, 0.0],
         [1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0]]
    r = sgt_signless_laplacian(A)
    Q = [list(map(float, row)) for row in r["Q"]]
    assert Q == [[1.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 1.0]]
    assert [float(d) for d in r["degree"]] == [1.0, 2.0, 1.0]
    assert r["n"] == 3
    assert float(r["m"]) == 2.0
    assert float(r["trace"]) == 4.0
    assert r["n_components"] == 1
    # a path is bipartite, so its single component is bipartite and Q is
    # singular with one zero eigenvalue per bipartite component
    assert r["bipartite_components"] == 1
    assert r["zero_eigenvalue_multiplicity"] == 1


def test_sgtsig_triangle_is_not_bipartite():
    """An odd cycle contributes no zero eigenvalue to Q."""
    A = [[0.0, 1.0, 1.0],
         [1.0, 0.0, 1.0],
         [1.0, 1.0, 0.0]]
    r = sgt_signless_laplacian(A)
    assert float(r["trace"]) == 6.0
    assert float(r["m"]) == 3.0
    assert r["bipartite_components"] == 0
    assert r["zero_eigenvalue_multiplicity"] == 0


def test_sgtsig_edge():
    """Two disjoint edges: two components, both bipartite."""
    A = [[0.0, 1.0, 0.0, 0.0],
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, 1.0, 0.0]]
    r = sgt_signless_laplacian(A)
    assert r["n_components"] == 2
    assert r["bipartite_components"] == 2
    assert r["zero_eigenvalue_multiplicity"] == 2
    assert float(r["trace"]) == 4.0
