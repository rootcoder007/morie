"""Tests for gradds.gradient_descent."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.gradds import gradient_descent


def _make_f_and_grad(a, b):
    """Return (f, grad_f) for f(x) = 0.5*(a*x[0] + b)^2 (scalar quadratic in x[0])."""
    def f(x):
        return 0.5 * (a * x[0] + b) ** 2

    def grad_f(x):
        return [a * (a * x[0] + b)]

    return f, grad_f


def test_gradds_basic():
    """Test basic functionality on a 1D quadratic f(x) = 0.5*(a*x + b)^2."""
    f, grad_f = _make_f_and_grad(a=2.0, b=-4.0)
    x0 = [3.0]
    lr = 0.1
    steps = 50

    result = gradient_descent(f, grad_f, x0, lr, steps)

    # Documented keys returned by the function.
    assert "estimate" in result
    assert "x" in result
    assert "f_path" in result
    assert "grad_norm" in result
    assert "steps_used" in result
    assert "converged" in result
    assert "n" in result

    # x has the same length as x0.
    assert len(result["x"]) == len(x0)
    assert result["n"] == len(x0)

    # f_path starts with f(x0) and has length steps_used + 1.
    assert result["f_path"][0] == 0.5 * (2.0 * x0[0] + (-4.0)) ** 2
    assert len(result["f_path"]) == result["steps_used"] + 1

    # For a convex quadratic, the function value must not increase over the path.
    for i in range(1, len(result["f_path"])):
        assert result["f_path"][i] <= result["f_path"][i - 1] + 1e-12

    # The reported estimate equals f(x) at the final iterate (independent calc).
    final_x = result["x"]
    expected_estimate = 0.5 * (2.0 * final_x[0] + (-4.0)) ** 2
    assert result["estimate"] == expected_estimate

    # steps_used is at most the requested steps and is a non-negative int.
    assert 0 <= result["steps_used"] <= steps


def test_gradds_edge():
    """Test edge cases: zero initial gradient and arithmetic checks."""
    # Start exactly at the minimiser so the first gradient norm is 0.
    f, grad_f = _make_f_and_grad(a=3.0, b=-6.0)  # minimiser at x = 2.0
    x0 = [2.0]
    lr = 0.1
    steps = 10

    result = gradient_descent(f, grad_f, x0, lr, steps)

    # Independent computation of f at x0.
    assert result["f_path"][0] == 0.5 * (3.0 * 2.0 + (-6.0)) ** 2  # 0.0

    # With zero initial gradient we should converge on the very first check.
    assert result["converged"] is True
    assert result["steps_used"] == 0
    assert result["grad_norm"] == 0.0
    assert result["x"] == [2.0]
    assert result["estimate"] == 0.0
