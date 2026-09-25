"""Tests for bsaclass.rangayyan_svm_kernel (sec. 10.4.5)."""

import math

import pytest

from morie.fn.bsaclass import rangayyan_svm_kernel


X = [[1, 2], [2, 3], [3, 3.5], [-1, -0.5], [-2, -1.5], [0, -1]]
Y = [1, 1, 1, -1, -1, -1]


def _k(u, v, g):
    return math.exp(-g * sum((a - b) ** 2 for a, b in zip(u, v)))


def test_rgsvmk_basic():
    """The decision at a query is sum_i a_i y_i K(x_i, q) + b with the
    RBF kernel (gamma defaulting to 1/n_features), and its sign is the
    assigned class; the dual constraint sum a y = 0 holds."""
    q = [0.5, 0.5]
    r = rangayyan_svm_kernel(X, Y, query=q, C=10.0, tol=1e-9, maxiter=20000)
    a, b, g = [float(v) for v in r["alpha"]], float(r["b"]), float(r["gamma"])
    assert g == pytest.approx(0.5, abs=1e-15)
    dec = sum(a[i] * Y[i] * _k(X[i], q, g) for i in range(6)) + b
    assert float(r["decision"]) == pytest.approx(dec, abs=1e-12)
    assert r["assigned"] == (1 if dec >= 0 else -1)
    assert sum(ai * yi for ai, yi in zip(a, Y)) == pytest.approx(0.0, abs=1e-9)
    assert all(-1e-12 <= ai <= 10.0 + 1e-12 for ai in a)


def test_rgsvmk_edge():
    """Unknown kernels and bad labels raise."""
    with pytest.raises(ValueError):
        rangayyan_svm_kernel(X, Y, kernel="laplace")
    with pytest.raises(ValueError):
        rangayyan_svm_kernel(X, [0, 1, 1, 0, 0, 1])
