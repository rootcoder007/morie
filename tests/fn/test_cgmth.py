"""Tests for cgmth (conjugate gradient minimisation)."""

import numpy as np
import pytest

from morie.fn.cgmth import cgmth


def test_cgmth_finds_the_minimum_of_a_shifted_quadratic():
    """The docstring's own example: minimum of (x-2)^2 + (y-3)^2 is (2, 3)."""
    f = lambda x: (x[0] - 2) ** 2 + (x[1] - 3) ** 2
    gf = lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 3)])
    x = cgmth(f, gf, np.zeros(2))
    np.testing.assert_allclose(np.asarray(x, dtype=float), [2.0, 3.0], atol=1e-4)


def test_cgmth_solves_an_ill_conditioned_quadratic():
    """f(x) = 1/2 x'Ax - b'x with condition number 100: gradient descent
    crawls here; conjugate directions do not. The minimiser is A^{-1} b,
    known in closed form."""
    A = np.diag([1.0, 100.0])
    b = np.array([1.0, 1.0])
    f = lambda x: 0.5 * x @ A @ x - b @ x
    gf = lambda x: A @ x - b
    x = np.asarray(cgmth(f, gf, np.zeros(2), max_iter=500), dtype=float)
    np.testing.assert_allclose(x, np.linalg.solve(A, b), atol=1e-3)


def test_cgmth_full_output_reports_convergence():
    f = lambda x: float(x @ x)
    gf = lambda x: 2 * x
    x, info = cgmth(f, gf, np.array([5.0, -3.0]), full_output=True)
    assert info["converged"]
    assert info["final_value"] == pytest.approx(0.0, abs=1e-8)
    assert info["iterations"] >= 0


def test_cgmth_starting_at_the_minimum_stays_there():
    f = lambda x: float((x - 1) @ (x - 1))
    gf = lambda x: 2 * (x - 1)
    x = np.asarray(cgmth(f, gf, np.ones(3)), dtype=float)
    np.testing.assert_allclose(x, 1.0, atol=1e-10)
