"""Tests for sgtmodq.sgt_modularity_q."""

from morie.fn.sgtmodq import sgt_modularity_q


def _two_triangles_bridged():
    """Triangles {0,1,2} and {3,4,5} joined by the single edge 2-3."""
    A = [[0.0] * 6 for _ in range(6)]
    for i, j in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)]:
        A[i][j] = A[j][i] = 1.0
    return A


def _modularity(A, labels):
    """Q = (1/2m) sum_ij (A_ij - k_i k_j / 2m) delta(c_i, c_j)."""
    n = len(A)
    k = [sum(A[i]) for i in range(n)]
    two_m = sum(k)
    return sum(
        A[i][j] - k[i] * k[j] / two_m
        for i in range(n)
        for j in range(n)
        if labels[i] == labels[j]
    ) / two_m


def test_sgtmodq_basic():
    """The natural partition of two bridged triangles scores 5/14."""
    A = _two_triangles_bridged()
    labels = [0, 0, 0, 1, 1, 1]
    r = sgt_modularity_q(A, labels)
    assert abs(r["Q"] - 5.0 / 14.0) < 1e-9
    assert abs(r["Q"] - _modularity(A, labels)) < 1e-9
    assert r["estimate"] == r["Q"]
    assert r["n_communities"] == 2
    assert r["n"] == 6


def test_sgtmodq_one_community_scores_zero():
    """One community holding every node gives exactly zero."""
    A = _two_triangles_bridged()
    r = sgt_modularity_q(A, [0] * 6)
    assert abs(r["Q"]) < 1e-9
    assert r["n_communities"] == 1


def test_sgtmodq_edge():
    """Singleton communities leave only the null term, so Q < 0."""
    A = _two_triangles_bridged()
    labels = list(range(6))
    r = sgt_modularity_q(A, labels)
    assert r["n_communities"] == 6
    assert abs(r["Q"] - _modularity(A, labels)) < 1e-9
    assert r["Q"] < 0.0
