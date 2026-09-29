"""Tests for piepar.pie_parameters."""

import math

import pytest

from morie.fn.piepar import pie_parameters

Y = [2.1, 3.4, 1.9, 5.6, 2.8, 3.1, 6.2, 2.5, 3.3, 4.7, 5.1, 2.2]
X = [
    [1.0, 0.3],
    [2.0, 1.1],
    [0.5, 0.2],
    [3.5, 0.9],
    [1.5, 1.4],
    [2.2, 0.1],
    [4.1, 0.6],
    [1.2, 0.8],
    [2.0, 0.5],
    [3.0, 1.2],
    [3.8, 0.4],
    [0.9, 1.0],
]
XS = [1.0, 1.5, 2.0]


def _ols(W, y):
    """Normal equations by Gauss-Jordan, independent of the module's QR."""
    p = len(W[0])
    A = [[sum(r[i] * r[j] for r in W) for j in range(p)] + [float(i == j) for j in range(p)] for i in range(p)]
    for c in range(p):
        piv = A[c][c]
        A[c] = [v / piv for v in A[c]]
        for r in range(p):
            if r != c:
                f = A[r][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    inv = [row[p:] for row in A]
    xty = [sum(W[k][i] * y[k] for k in range(len(y))) for i in range(p)]
    b = [sum(inv[i][j] * xty[j] for j in range(p)) for i in range(p)]
    return b, inv


def test_piepar_is_the_shifted_exposure_contrast():
    W = [[1.0] + r for r in X]
    b, inv = _ols(W, Y)
    n, p = len(Y), 3
    xbar = sum(r[0] for r in X) / n
    dx = sum(XS) / 3 - xbar
    rss = sum((Y[i] - sum(W[i][j] * b[j] for j in range(p))) ** 2 for i in range(n))
    r = pie_parameters(Y, X, XS)
    assert r["estimate"] == pytest.approx(b[1] * dx, rel=1e-10)
    assert r["se"] == pytest.approx(abs(dx) * math.sqrt(rss / (n - p) * inv[1][1]), rel=1e-10)
    assert r["observed"] == pytest.approx(sum(Y) / n, rel=1e-14)


def test_piepar_edge():
    """Intervening at the observed mean exposure changes nothing."""
    xbar = sum(r[0] for r in X) / len(X)
    r = pie_parameters(Y, X, [xbar])
    assert r["estimate"] == pytest.approx(0.0, abs=1e-12)
    assert r["se"] == pytest.approx(0.0, abs=1e-12)
