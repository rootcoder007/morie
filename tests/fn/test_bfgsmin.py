"""Tests for bfgsmin.bfgs_minimize (Nocedal-Wright Alg 6.1)."""

import pytest

from morie.fn.bfgsmin import bfgs_minimize


def test_bfgs_finds_the_quadratic_minimiser():
    """f = 1/2 x'Ax - b'x has its minimum at A^-1 b."""
    A = [[3.0, 1.0], [1.0, 2.0]]
    b = [1.0, -1.0]
    det = A[0][0] * A[1][1] - A[0][1] ** 2
    xstar = [(A[1][1] * b[0] - A[0][1] * b[1]) / det, (A[0][0] * b[1] - A[0][1] * b[0]) / det]

    def f(x):
        return 0.5 * sum(x[i] * A[i][j] * x[j] for i in range(2) for j in range(2)) - b[0] * x[0] - b[1] * x[1]

    def g(x):
        return [sum(A[i][j] * x[j] for j in range(2)) - b[i] for i in range(2)]

    r = bfgs_minimize(f, [2.0, 2.0], grad=g)
    assert r["converged"]
    assert r["x"] == pytest.approx(xstar, abs=1e-8)
    assert r["fun"] == pytest.approx(f(xstar), abs=1e-12)


def test_bfgs_stationary_start():
    r = bfgs_minimize(lambda x: (x[0] - 1.0) ** 2, [1.0])
    assert r["n_iter"] == 0 and r["x"] == [1.0]
