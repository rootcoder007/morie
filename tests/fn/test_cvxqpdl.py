"""Tests for cvxqpdl.boyd_qp_dual."""

import math

from morie.fn import _array_core as np

from morie.fn.cvxqpdl import boyd_qp_dual


def test_cvxqpdl_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 5
    m = 3

    # Build a positive definite Hessian via A A^T + I
    A = rng.normal(0, 1, (n, n))
    P = A @ A.T + np.eye(n)

    q = rng.normal(0, 1, n)
    G = rng.normal(0, 1, (m, n))
    h = rng.normal(0, 1, m)

    result = boyd_qp_dual(P, q, G, h)

    assert isinstance(result, dict)
    for key in ("lambda_", "dual_value", "x", "primal_value", "gap",
                "strong_duality", "slack", "active", "complementary_slackness"):
        assert key in result

    # Strong duality holds for affine constraints (Slater's condition).
    assert bool(result["strong_duality"])
    assert math.isfinite(float(result["primal_value"]))
    assert math.isfinite(float(result["dual_value"]))
    assert math.isfinite(float(result["gap"]))

    # Shape checks on returned vectors.
    assert len(result["x"]) == n
    assert len(result["lambda_"]) == m
    assert len(result["slack"]) == m
    assert len(result["active"]) == m
    # complementary_slackness is a single boolean flag (λ_i * s_i = 0 for all i).
    assert isinstance(result["complementary_slackness"], bool)
    assert bool(result["complementary_slackness"])


def test_cvxqpdl_edge():
    """Test edge cases."""
    # Minimal 2-variable, 1-constraint problem from the docstring:
    # min |x|^2/2 - x1 - x2 s.t. x1 + x2 <= 1.
    P = np.eye(2)
    q = [-1.0, -1.0]
    G = [[1.0, 1.0]]
    h = [1.0]

    result = boyd_qp_dual(P, q, G, h)

    assert isinstance(result, dict)
    for key in ("lambda_", "dual_value", "x", "primal_value", "gap",
                "strong_duality", "slack", "active", "complementary_slackness"):
        assert key in result

    assert bool(result["strong_duality"])
    assert math.isfinite(float(result["primal_value"]))
    assert math.isfinite(float(result["dual_value"]))
    assert math.isfinite(float(result["gap"]))

    # The single constraint is binding at the optimum.
    assert bool(result["active"][0])
    # Primal and dual must agree (no duality gap).
    assert abs(float(result["primal_value"]) - float(result["dual_value"])) < 1e-6
    # x has length 2, lambda_ has length 1.
    assert len(result["x"]) == 2
    assert len(result["lambda_"]) == 1
    assert len(result["slack"]) == 1
    # complementary_slackness is a single boolean flag.
    assert isinstance(result["complementary_slackness"], bool)
    assert bool(result["complementary_slackness"])
