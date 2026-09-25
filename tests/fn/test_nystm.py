"""Tests for nystm.nystrom_approximation."""

import math

import pytest

from morie.fn.nystm import nystrom_approximation

X = [[1.0, 0.5], [0.2, -1.0], [2.0, 1.5], [-0.7, 0.3], [0.4, 0.4], [1.1, -0.6]]


def test_nystm_basic():
    """Q = K_nm K_mm^-1 K_mn with Gaussian kernel exp(-gamma |x - y|^2)
    and landmarks rows 1 and 3 (1-based), recomputed with the 2 x 2
    inverse written out."""
    g = 0.5
    k = lambda a, b: math.exp(-g * sum((u - v) ** 2 for u, v in zip(a, b)))
    L = [X[0], X[2]]
    Kmm = [[k(a, b) for b in L] for a in L]
    det = Kmm[0][0] * Kmm[1][1] - Kmm[0][1] * Kmm[1][0]
    inv = [[Kmm[1][1] / det, -Kmm[0][1] / det], [-Kmm[1][0] / det, Kmm[0][0] / det]]
    Knm = [[k(x, l) for l in L] for x in X]
    Q = [[sum(Knm[i][a] * inv[a][b] * Knm[j][b] for a in range(2) for b in range(2))
          for j in range(6)] for i in range(6)]
    r = nystrom_approximation(X, [1, 3], kernel="gaussian", gamma=g)
    assert isinstance(r, dict)
    for i in range(6):
        assert list(r["Q"][i]) == pytest.approx(Q[i], rel=1e-10, abs=1e-12)
    # the landmark block is reproduced exactly
    assert r["Q"][0][2] == pytest.approx(Kmm[0][1], rel=1e-12)


def test_nystm_edge():
    """The linear kernel is X X^T / p (Cuevas et al. 2020); two
    independent landmark rows span the rank-2 row space, so Q = K
    exactly; 0 is not a 1-based index."""
    r = nystrom_approximation(X, [1, 2], kernel="linear")
    for i in range(6):
        assert list(r["Q"][i]) == pytest.approx(
            [sum(a * b for a, b in zip(X[i], X[j])) / 2 for j in range(6)], rel=1e-10, abs=1e-12)
    with pytest.raises(ValueError):
        nystrom_approximation(X, [0, 1])
