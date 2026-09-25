"""Tests for sgcL.sgc (simplified graph convolution, Wu et al. 2019)."""

import math

import pytest

from morie.fn.sgcL import sgc

A = [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]]
X = [[1.0, 0.0], [0.0, 2.0], [0.5, 0.5], [3.0, -1.0]]


def _S():
    At = [[A[i][j] + (1 if i == j else 0) for j in range(4)] for i in range(4)]
    d = [sum(r) for r in At]
    return [[At[i][j] / math.sqrt(d[i] * d[j]) for j in range(4)] for i in range(4)]


def _mm(M, Y):
    return [[sum(M[i][k] * Y[k][j] for k in range(len(Y))) for j in range(len(Y[0]))]
            for i in range(len(M))]


def test_sgcL_basic():
    """S^K X with S = D~^-1/2 (A + I) D~^-1/2, recomputed for K = 2."""
    S = _S()
    exp = _mm(S, _mm(S, X))
    r = sgc(A, X, 2)
    out = r["X"]
    for a, b in zip(out, exp):
        assert list(a) == pytest.approx(b, rel=1e-13)


def test_sgcL_edge():
    """K = 0 returns X unchanged; an isolated vertex still has the
    self-loop degree 1, so it keeps its own features."""
    r = sgc(A, X, 0)
    out = r["X"]
    assert [list(v) for v in out] == X
    A2 = [[0, 0], [0, 0]]
    r2 = sgc(A2, [[2.0], [5.0]], 3)
    out2 = r2["X"]
    assert [list(v) for v in out2] == [[2.0], [5.0]]
