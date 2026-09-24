"""Tests for newraf.newton_raphson."""

from morie.fn import _array_core as np

import math
import pytest

from morie.fn.newraf import newton_raphson


def test_newraf_basic():
    """Test basic functionality on a positive-definite diagonal quadratic."""
    p = 5
    rng = np.random.default_rng(0)
    d = [abs(float(v)) + 0.5 for v in rng.normal(0, 1, p)]
    b = list(rng.normal(0, 1, p))

    def f(x):
        return sum(d[i] * x[i] * x[i] / 2 - b[i] * x[i] for i in range(p))

    def grad_f(x):
        return [d[i] * x[i] - b[i] for i in range(p)]

    def hess_f(x):
        return [[d[i] if i == j else 0.0 for j in range(p)] for i in range(p)]

    x0 = list(rng.normal(0, 1, p))
    result = newton_raphson(f, grad_f, hess_f, x0)

    assert isinstance(result, dict)
    assert "x" in result
    assert "estimate" in result
    assert "fval" in result
    assert "grad_norm" in result
    assert "iterations" in result
    assert "converged" in result
    assert "p" in result
    assert "method" in result
    assert result["p"] == p
    assert len(result["x"]) == p
    assert isinstance(result["iterations"], int)
    assert result["iterations"] >= 0
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["grad_norm"])
    assert result["grad_norm"] >= 0.0
    # For a quadratic with constant PD Hessian, Newton reaches the exact
    # minimum in one step, so the gradient norm must fall below tol.
    assert result["converged"] == 1.0


def test_newraf_edge():
    """Test edge case: an empty starting point is rejected by the solver."""
    def f(x):
        return sum(v * v for v in x)

    def grad_f(x):
        return [2.0 * v for v in x]

    def hess_f(x):
        n = len(x)
        return [[2.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    with pytest.raises(ValueError):
        newton_raphson(f, grad_f, hess_f, [])
