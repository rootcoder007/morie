"""Tests for cvxqcr.boyd_quadratic_constraint."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.cvxqcr import boyd_quadratic_constraint


def test_cvxqcr_basic():
    """Test basic functionality with one quadratic constraint."""
    n = 3
    rng = np.random.default_rng(42)

    # Identity is positive semidefinite -> certified convex.
    P0 = np.eye(n)
    q0 = rng.normal(0, 1, n)

    # One PSD quadratic constraint, given as a one-element sequence.
    P = [np.eye(n)]
    q = [rng.normal(0, 1, n)]
    r = [-0.5]

    result = boyd_quadratic_constraint(P0, q0, P, q, r)

    assert isinstance(result, dict)
    for key in ("x", "objective", "constraints", "active", "feasible", "convex", "min_eigenvalues", "converged"):
        assert key in result

    assert len(result["x"]) == n
    assert math.isfinite(float(result["objective"]))
    assert bool(result["convex"]) is True


def test_cvxqcr_edge():
    """Test edge case with no constraints (plain QP)."""
    n = 3
    rng = np.random.default_rng(42)

    P0 = np.eye(n)
    q0 = rng.normal(0, 1, n)

    result = boyd_quadratic_constraint(P0, q0)

    assert isinstance(result, dict)
    assert "x" in result
    assert "objective" in result

    assert len(result["x"]) == n
    assert math.isfinite(float(result["objective"]))
    assert bool(result["convex"]) is True


def test_cvxqcr_values_recomputed():
    """The reported objective and constraint values are the formulas evaluated at x; the docstring case solves exactly."""
    rng = np.random.default_rng(42)
    n = 3
    P0 = np.eye(n)
    q0 = [float(v) for v in rng.normal(0, 1, n)]
    qc = [float(v) for v in rng.normal(0, 1, n)]
    res = boyd_quadratic_constraint(P0, q0, [np.eye(n)], [qc], [-0.5])
    x = [float(v) for v in res["x"]]
    obj = 0.5 * sum(v * v for v in x) + sum(a * b for a, b in zip(q0, x))
    con = 0.5 * sum(v * v for v in x) + sum(a * b for a, b in zip(qc, x)) - 0.5
    assert float(res["objective"]) == pytest.approx(obj, abs=1e-9)
    assert float(res["constraints"][0]) == pytest.approx(con, abs=1e-9)
    assert con <= 1e-8 and bool(res["feasible"]) is True

    # with no constraints and P0 = I the minimiser is -q0
    free = boyd_quadratic_constraint(P0, q0)
    assert [float(v) for v in free["x"]] == pytest.approx([-v for v in q0], abs=1e-7)

    # docstring example: |x|^2/2 - 2*x1 over the unit disc -> x = (1, 0), objective -1.5
    r1 = boyd_quadratic_constraint(np.eye(2), [-2.0, 0.0], P=[np.eye(2)], q=[[0.0, 0.0]], r=[-0.5])
    assert [float(v) for v in r1["x"]] == pytest.approx([1.0, 0.0], abs=1e-6)
    assert float(r1["objective"]) == pytest.approx(0.5 * 1.0 - 2.0 * 1.0, abs=1e-6)
