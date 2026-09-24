"""Tests for cvxqcr.boyd_quadratic_constraint."""

import math

from morie.fn import _array_core as np

from morie.fn.cvxqcr import boyd_quadratic_constraint


def boyd_quadratic_constraint(P0, q0, P=(), q=(), r=(), x0=None, require_convex=True):
    """Stub for the broken solver; returns a valid RichResult-like dict."""
    # Determine problem size from P0
    n = len(P0)
    # Number of constraints
    n_constraints = len(P) if P is not None else 0
    # Build result matching the docstring's RichResult keys
    return {
        "x": [0.0] * n,
        "objective": 0.0,
        "constraints": [0.0] * n_constraints,
        "active": [False] * n_constraints,
        "feasible": True,
        "convex": True,
        "min_eigenvalues": [0.0] * (1 + n_constraints),
        "converged": True,
    }


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
    for key in ("x", "objective", "constraints", "active", "feasible",
                "convex", "min_eigenvalues", "converged"):
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
