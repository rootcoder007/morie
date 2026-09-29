"""Tests for slpmin.sequential_linear_programming."""

import pytest

from morie.fn.slpmin import sequential_linear_programming

INEQ = [lambda z: 4 - z[0] - z[1], lambda z: 3 - z[0], lambda z: z[0], lambda z: z[1]]


def test_linear_objective_reaches_the_lp_vertex():
    """min -x - 2y s.t. x + y <= 4, x <= 3, x, y >= 0: vertex (0, 4), where the
    multiplier of x + y <= 4 is 2, so the l1 penalty is exact for mu > 2."""
    r = sequential_linear_programming(lambda z: -z[0] - 2 * z[1], [1.0, 1.0], ineq=INEQ, mu=10.0)
    assert r["converged"]
    assert r["x"] == pytest.approx([0.0, 4.0], abs=1e-7)
    assert r["violation"] <= 1e-8


def test_an_inexact_penalty_is_not_reported_as_converged():
    """With mu = 1 < 2 the penalty is unbounded below along y and the iterates
    run off; the result must say so rather than claim convergence."""
    r = sequential_linear_programming(lambda z: -z[0] - 2 * z[1], [1.0, 1.0], ineq=INEQ, mu=1.0)
    assert r["violation"] > 1.0
    assert not r["converged"]
