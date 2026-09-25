"""Tests for specClust.spectral_clustering (Ng, Jordan & Weiss 2001)."""

import math

import pytest

from morie.fn.specClust import spectral_clustering


def _two_cliques(m=4, bridge=0.1):
    n = 2 * m
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and (i < m) == (j < m):
                A[i][j] = 1.0
    A[m - 1][m] = A[m][m - 1] = bridge
    return A


def test_specClust_basic():
    """Two dense blocks joined by one weak edge split along the Fiedler
    vector of L_sym = I - D^-1/2 A D^-1/2; the smallest eigenvalue of
    L_sym is exactly 0 with eigenvector D^1/2 1, and the second solves
    L_sym v = lambda v."""
    A = _two_cliques()
    r = spectral_clustering(A, k=2)
    lab = r["labels"]
    assert len(set(lab[:4])) == 1 and len(set(lab[4:])) == 1 and lab[0] != lab[4]
    assert r["sizes"] == [4.0, 4.0]
    d = [sum(row) for row in A]
    assert r["degree"] == pytest.approx(d, abs=1e-15)
    lam0, lam1 = r["eigenvalues"][:2]
    assert lam0 == pytest.approx(0.0, abs=1e-10)
    f = r["fiedler"]
    n = len(A)
    Lf = [f[i] - sum(A[i][j] * f[j] / math.sqrt(d[i] * d[j]) for j in range(n)) for i in range(n)]
    assert Lf == pytest.approx([lam1 * t for t in f], abs=1e-9)
    # the Fiedler vector is orthogonal to D^1/2 1
    assert sum(math.sqrt(d[i]) * f[i] for i in range(n)) == pytest.approx(0.0, abs=1e-9)


def test_specClust_edge():
    """k out of range, a non-zero diagonal, an isolated node and an
    asymmetric matrix all raise."""
    A = _two_cliques()
    with pytest.raises(ValueError):
        spectral_clustering(A, k=1)
    B = [r[:] for r in A]
    B[0][0] = 1.0
    with pytest.raises(ValueError):
        spectral_clustering(B)
    C = [r[:] + [0.0] for r in A] + [[0.0] * 9]
    with pytest.raises(ValueError):
        spectral_clustering(C)
    E = [r[:] for r in A]
    E[0][1] = 2.0
    with pytest.raises(ValueError):
        spectral_clustering(E)
