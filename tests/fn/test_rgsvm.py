"""Tests for bsaclass.rangayyan_svm (linear SVM, sec. 10.4.5)."""

import math

import pytest

from morie.fn.bsaclass import rangayyan_svm


X = [[1, 2], [2, 3], [3, 3.5], [-1, -0.5], [-2, -1.5], [0, -1]]
Y = [1, 1, 1, -1, -1, -1]


def test_rgsvm_basic():
    """Hard-margin KKT on separable data: y(w'x + b) >= 1 everywhere,
    = 1 on the support vectors, w = sum a y x, sum a y = 0, and the
    margin is 2/|w|."""
    r = rangayyan_svm(X, Y, C=1e4, tol=1e-9, maxiter=20000)
    w, b, a = [float(v) for v in r["w"]], float(r["b"]), [float(v) for v in r["alpha"]]
    tol = 1e-5
    for xi, yi in zip(X, Y):
        assert yi * (w[0] * xi[0] + w[1] * xi[1] + b) >= 1 - tol
    for i in r["support_vectors"]:
        assert Y[i] * (w[0] * X[i][0] + w[1] * X[i][1] + b) == pytest.approx(1.0, abs=tol)
    assert w == pytest.approx([sum(a[i] * Y[i] * X[i][j] for i in range(6)) for j in range(2)], abs=1e-9)
    assert sum(ai * yi for ai, yi in zip(a, Y)) == pytest.approx(0.0, abs=1e-9)
    assert r["margin"] == pytest.approx(2 / math.hypot(*w), rel=1e-9)
    assert r["training_accuracy"] == 1.0


def test_rgsvm_edge():
    """Two points: w = (0.5, 0.5), b = 0 (max-margin bisector); labels
    other than +-1 raise."""
    r = rangayyan_svm([[-1.0, -1.0], [1.0, 1.0]], [-1, 1], C=1e4, tol=1e-12)
    assert [float(v) for v in r["w"]] == pytest.approx([0.5, 0.5], abs=1e-8)
    assert float(r["b"]) == pytest.approx(0.0, abs=1e-8)
    with pytest.raises(ValueError):
        rangayyan_svm(X, [0, 1, 1, 0, 0, 1])
