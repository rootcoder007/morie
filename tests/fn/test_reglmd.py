"""Tests for reglmd.regression_estimator (multivariate regression estimator)."""

import pytest

from morie.fn import _array_core as np
from morie.fn.reglmd import regression_estimator

Y = [4.1, 5.0, 3.2, 6.8, 5.5, 4.4, 7.1, 6.0]
X = [[1.0, 0.2], [1.5, 0.1], [0.8, 0.5], [2.2, 0.3], [1.7, 0.9], [1.1, 0.4], [2.5, 0.6], [2.0, 0.2]]
XM = [1.6, 0.35]


def _solve2(A, b):
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    return [(b[0] * A[1][1] - A[0][1] * b[1]) / det, (A[0][0] * b[1] - b[0] * A[1][0]) / det]


def test_reglmd_basic():
    """ybar + b'(Xbar - xbar) with b the centred least-squares slopes,
    and R^2 = explained / total sum of squares, recomputed."""
    n = 8
    xb = [sum(r[j] for r in X) / n for j in range(2)]
    yb = sum(Y) / n
    Xc = [[r[j] - xb[j] for j in range(2)] for r in X]
    yc = [v - yb for v in Y]
    A = [[sum(r[i] * r[j] for r in Xc) for j in range(2)] for i in range(2)]
    b = _solve2(A, [sum(r[i] * v for r, v in zip(Xc, yc)) for i in range(2)])
    r = regression_estimator(Y, X, XM)
    assert isinstance(r, dict)
    assert [float(v) for v in r["coefficients"]] == pytest.approx(b, rel=1e-12)
    assert r["mean"] == pytest.approx(yb + b[0] * (XM[0] - xb[0]) + b[1] * (XM[1] - xb[1]), rel=1e-13)
    fit = [b[0] * u + b[1] * w for u, w in Xc]
    assert r["R2"] == pytest.approx(sum(f * f for f in fit) / sum(v * v for v in yc), rel=1e-12)


def test_reglmd_edge():
    """When the sample means equal the population means there is no
    adjustment; a mean vector of the wrong length is refused."""
    xb = [sum(r[j] for r in X) / 8 for j in range(2)]
    r = regression_estimator(Y, X, xb)
    assert r["adjustment"] == pytest.approx(0.0, abs=1e-14)
    assert r["mean"] == pytest.approx(sum(Y) / 8, rel=1e-14)
    with pytest.raises(ValueError):
        regression_estimator(Y, X, [1.0, 2.0, 3.0])
