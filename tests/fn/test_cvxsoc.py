"""Tests for cvxsoc.boyd_socp (Boyd and Vandenberghe sec. 4.4.2)."""

import math

import pytest

from morie.fn.cvxsoc import boyd_socp


I2 = [[1.0, 0.0], [0.0, 1.0]]
# the SQP solver stops at its 1e-6 constraint tolerance (the same tolr
# the function uses to flag active cones), so the optimum is checked to
# that tolerance rather than to rounding
TOL = 1e-6


def test_cvxsoc_basic():
    """min x1 over the unit disc is (-1, 0) with the cone active; min -x1
    over |x - (2, 0)| <= 1 is the far edge (3, 0).  The reported norms,
    right-hand sides and slacks are recomputed from x."""
    r = boyd_socp([1.0, 0.0], [I2], [[0.0, 0.0]], [[0.0, 0.0]], [1.0])
    x = [float(v) for v in r["x"]]
    assert x == pytest.approx([-1.0, 0.0], abs=TOL)
    assert r["objective"] == pytest.approx(x[0], abs=1e-15)
    assert float(r["lhs"][0]) == pytest.approx(math.hypot(*x), rel=1e-12)
    assert float(r["slack"][0]) == pytest.approx(float(r["rhs"][0]) - float(r["lhs"][0]), abs=1e-15)
    assert [bool(a) for a in r["active"]] == [True]
    assert r["feasible"] and r["converged"]
    s = boyd_socp([-1.0, 0.0], [I2], [[-2.0, 0.0]], [[0.0, 0.0]], [1.0])
    assert [float(v) for v in s["x"]] == pytest.approx([3.0, 0.0], abs=TOL)


def test_cvxsoc_edge():
    """Mismatched constraint lists raise."""
    with pytest.raises(ValueError):
        boyd_socp([1.0, 0.0], [I2], [[0.0, 0.0]], [[0.0, 0.0]], [1.0, 2.0])
