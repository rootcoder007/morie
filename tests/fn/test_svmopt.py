"""Tests for svmopt.svm_dual (SMO on the maximal violating pair)."""

import math

import pytest

from morie.fn.svmopt import kernel_matrix, svm_dual


def _data(n=20):
    X = [[math.sin(1.3 * k) + 0.4 * (1 if k % 2 else -1), math.cos(0.7 * k)] for k in range(n)]
    y = [1.0 if k % 2 else -1.0 for k in range(n)]
    return X, y


def test_svmopt_basic():
    """At the solution the KKT conditions of the soft-margin dual hold to
    the stopping gap: sum y a = 0, 0 <= a <= C, and with
    f(x_i) = sum_j a_j y_j K_ij + b, y_i f = 1 on free vectors, >= 1 at
    a = 0 and <= 1 at a = C.  (sklearn SVC(kernel='precomputed') agrees
    on these data to its own tolerance.)"""
    X, y = _data()
    K = kernel_matrix(X, "rbf", gamma=0.5)
    for i in range(len(X)):
        for j in range(len(X)):
            assert K[i][j] == pytest.approx(math.exp(-0.5 * math.dist(X[i], X[j]) ** 2), abs=1e-15)
    C = 2.0
    r = svm_dual(y, K, C=C)
    a, b = r["alpha"], r["b"]
    assert r["converged"]
    assert abs(sum(ai * yi for ai, yi in zip(a, y))) < 1e-12
    tol = 1e-7
    for i in range(len(y)):
        m = y[i] * (sum(a[j] * y[j] * K[i][j] for j in range(len(y))) + b)
        assert -1e-12 <= a[i] <= C + 1e-12
        if a[i] < 1e-10:
            assert m >= 1 - tol
        elif a[i] > C - 1e-10:
            assert m <= 1 + tol
        else:
            assert m == pytest.approx(1.0, abs=tol)
    obj = sum(a) - 0.5 * sum(a[i] * a[j] * y[i] * y[j] * K[i][j] for i in range(20) for j in range(20))
    assert r["objective"] == pytest.approx(obj, rel=1e-12)


def test_svmopt_edge():
    """Two points at -1 and +1 on a line (linear kernel): the maximum
    margin solution is a = (1/2, 1/2), b = 0; labels outside {-1, +1},
    a mis-sized kernel and C <= 0 raise."""
    r = svm_dual([-1.0, 1.0], [[1.0, -1.0], [-1.0, 1.0]], C=10.0)
    assert r["alpha"] == pytest.approx([0.5, 0.5], abs=1e-12)
    assert r["b"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        svm_dual([0.0, 1.0], [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        svm_dual([-1.0, 1.0], [[1.0]])
    with pytest.raises(ValueError):
        svm_dual([-1.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], C=0.0)
