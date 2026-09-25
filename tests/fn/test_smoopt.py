"""Tests for smoopt.smo_solver (Platt 1998 sequential minimal optimisation)."""

import pytest

from morie.fn.smoopt import smo_solver

Y = [1, 1, -1, -1]
X = [[2.0, 1.0], [1.5, 2.0], [-1.0, -0.5], [-2.0, 0.5]]
K = [[sum(a * b for a, b in zip(u, v)) for v in X] for u in X]


def test_smoopt_basic():
    """Separable data, large C: the hard-margin solution is analytic.
    The closest opposite pair x0, x2 are the only support vectors, with
    alpha = 2 / |x0 - x2|^2 and w = alpha (x0 - x2); b makes
    f(x) = w.x - b equal +-1 on them (Platt's sign convention)."""
    r = smo_solver(Y, K, C=10.0, tol=1e-6)
    d2 = sum((a - b) ** 2 for a, b in zip(X[0], X[2]))
    a = 2.0 / d2
    assert r["alpha"] == pytest.approx([a, 0.0, a, 0.0], abs=1e-6)
    w = [a * (p - q) for p, q in zip(X[0], X[2])]
    assert r["b"] == pytest.approx(sum(u * v for u, v in zip(w, X[0])) - 1.0, abs=1e-6)
    assert abs(sum(al * y for al, y in zip(r["alpha"], Y))) < 1e-12


def test_smoopt_edge():
    """Labels other than +-1 are refused."""
    with pytest.raises(ValueError):
        smo_solver([0, 1, 1, 0], K)
