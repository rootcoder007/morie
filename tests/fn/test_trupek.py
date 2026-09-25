"""Tests for trupek.trust_region (basic trust-region method)."""

import pytest

from morie.fn.trupek import trust_region


def f(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def g(x):
    return [-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)]


def h(x):
    return [[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200.0]]


@pytest.mark.parametrize("sub", ["steihaug", "dogleg", "exact"])
def test_trupek_basic(sub):
    """Rosenbrock from (-1.2, 1): the minimiser is (1, 1) with f = 0, and
    every pass but the last is an accepted or rejected step."""
    r = trust_region(f, g, h, [-1.2, 1.0], subproblem=sub, max_iter=500)
    assert r["converged"]
    assert r["x"] == pytest.approx([1.0, 1.0], abs=1e-8)
    assert r["fval"] == pytest.approx(0.0, abs=1e-15)
    # the loop counts the pass that finds the gradient below gtol
    assert r["accepted"] + r["rejected"] + 1 == r["iterations"]


def test_trupek_edge():
    """On a convex quadratic the exact subproblem takes the Newton step
    once the radius allows it; the Cauchy variant still converges on it;
    an unknown subproblem raises."""
    q = lambda x: (x[0] - 3) ** 2 + 10 * (x[1] + 1) ** 2
    qg = lambda x: [2 * (x[0] - 3), 20 * (x[1] + 1)]
    qh = lambda x: [[2.0, 0.0], [0.0, 20.0]]
    r = trust_region(q, qg, qh, [0.0, 0.0], delta=10.0, subproblem="exact")
    assert r["x"] == pytest.approx([3.0, -1.0], abs=1e-12)
    assert r["iterations"] <= 3
    rc = trust_region(q, qg, qh, [0.0, 0.0], subproblem="cauchy", max_iter=2000)
    assert rc["x"] == pytest.approx([3.0, -1.0], abs=1e-6)
    with pytest.raises(ValueError):
        trust_region(q, qg, qh, [0.0, 0.0], subproblem="newton")
