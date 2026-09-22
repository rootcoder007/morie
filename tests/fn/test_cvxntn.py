"""Tests for cvxntn.boyd_newton."""

from morie.fn import _array_core as np

from morie.fn.cvxntn import boyd_newton


def test_cvxntn_basic():
    """Test basic functionality on a positive-definite quadratic."""
    rng = np.random.default_rng(42)
    n = 5
    # Symmetric positive-definite Hessian: A'A + n*I
    M = rng.normal(0, 1, (n, n))
    hess = M @ M.T + n * np.eye(n)
    x = rng.normal(0, 1, n)
    grad = hess @ x  # gradient of 0.5 x'Hx at x
    result = boyd_newton(grad, hess)
    assert isinstance(result, dict)
    assert "step" in result
    assert "decrement" in result
    assert "is_descent" in result
    assert "pd" in result
    # Pure Newton step on a quadratic lands at the minimiser (x = 0 here).
    expected_x_plus_step = x + result["step"]
    assert np.allclose(expected_x_plus_step, np.zeros(n), atol=1e-10)
    # Hessian is positive definite by construction.
    assert bool(result["pd"]) is True
    assert bool(result["is_descent"]) is True
    # Independent decrement computation: 0.5 * step' H step = 0 at the minimum.
    expected_decrement = float(np.sqrt(max(grad @ -result["step"], 0.0)))
    assert result["decrement"] == expected_decrement


def test_cvxntn_edge():
    """Test edge case: indefinite Hessian reported and not silently trusted."""
    # 1x1 indefinite Hessian: -1 (negative), gradient +1.
    result = boyd_newton(np.array([1.0]), np.array([[-1.0]]))
    assert isinstance(result, dict)
    assert bool(result["pd"]) is False
    assert "step" in result
    assert "decrement" in result
    assert "is_descent" in result
