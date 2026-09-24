"""Tests for cvxdgp.boyd_dual_problem."""

from morie.fn import _array_core as np

import math

from morie.fn.cvxdgp import boyd_dual_problem


def test_cvxdgp_basic():
    """Test basic functionality with the example from the docstring."""
    # Example: g(lam, nu) = -lam[0]^2 + lam[0] - 1.0, n_lambda=1, primal_value=-0.75
    g = lambda lam, nu: -lam[0] ** 2 + lam[0] - 1.0
    result = boyd_dual_problem(g, n_lambda=1, primal_value=-0.75, seed=42)

    # Assert that result is a dict-like RichResult
    assert isinstance(result, dict)

    # Check that required keys are present
    required_keys = {"lambda_", "nu", "dual_value", "active",
                      "bound_improves", "duality_gap", "strong_duality",
                      "concave", "converged"}
    for key in required_keys:
        assert key in result, f"Missing key {key} in result"

    # Check shapes: lambda_ length 1, nu length 0
    assert len(result["lambda_"]) == 1
    assert len(result["nu"]) == 0

    # Check that dual_value is close to -0.75 (the optimum)
    assert math.isfinite(result["dual_value"])
    assert abs(result["dual_value"] - (-0.75)) < 1e-6

    # Check that lambda_[0] is close to 0.5
    assert abs(result["lambda_"][0] - 0.5) < 1e-6

    # Check that duality_gap is close to 0.0 (since primal_value equals dual_value)
    assert math.isfinite(result["duality_gap"])
    assert abs(result["duality_gap"]) < 1e-6

    # Check that strong_duality is True
    assert result["strong_duality"] is True

    # Check that bound_improves is True (the optimum beats the free bound)
    assert result["bound_improves"] is True

    # Check that concave is True (g is concave)
    assert result["concave"] is True

    # Check that converged is True
    assert result["converged"] is True


def test_cvxdgp_edge():
    """Test edge case where unconstrained optimum is negative, so constraint binds."""
    g = lambda lam, nu: -lam[0] ** 2 - lam[0] - 1.0
    result = boyd_dual_problem(g, n_lambda=1, seed=42)

    assert isinstance(result, dict)

    # Check shapes
    assert len(result["lambda_"]) == 1
    assert len(result["nu"]) == 0

    # The optimum is at lambda=0 (since negative is infeasible)
    assert abs(result["lambda_"][0]) < 1e-6
    assert abs(result["dual_value"] - (-1.0)) < 1e-6

    # bound_improves should be False because the free bound (lambda=0) is already optimal
    assert result["bound_improves"] is False

    # g is concave
    assert result["concave"] is True

    # converged is a boolean (value not guaranteed to be True for boundary optima)
    assert isinstance(result["converged"], bool)
