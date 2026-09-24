"""Tests for prdldm.prox_method."""

import math

from morie.fn import _array_core as np
from morie.fn.prdldm import prox_method


def test_prdldm_basic():
    """Test basic functionality on a smooth convex quadratic."""
    def f(x):
        return 0.5 * sum(v * v for v in x)

    def grad_f(x):
        return [float(v) for v in x]

    def prox_g(y, lr):
        return [float(v) for v in y]

    x0 = [1.0, -2.0, 0.5, 3.0]
    lr = 0.1
    result = prox_method(f, grad_f, prox_g, x0, lr, n_iter=50)
    assert isinstance(result, dict)
    assert "x" in result
    assert "objective" in result
    assert "n_iter" in result
    assert "lr" in result
    assert "relaxation" in result
    assert "step_norm" in result
    assert "method" in result
    assert isinstance(result["x"], list)
    assert len(result["x"]) == len(x0)
    assert math.isfinite(result["objective"])
    assert result["n_iter"] == 50
    assert math.isclose(result["lr"], 0.1)
    assert math.isclose(result["relaxation"], 1.0)
    assert math.isfinite(result["step_norm"])
    assert isinstance(result["method"], str)


def test_prdldm_edge():
    """Test edge cases with a small input and non-default relaxation."""
    def f(x):
        return sum(v * v for v in x)

    def grad_f(x):
        return [2.0 * v for v in x]

    def prox_g(y, lr):
        # Non-trivial proximity operator (soft-thresholding) on g(x) = t * |x|_1
        t = 0.05
        return [math.copysign(max(abs(v) - t, 0.0), v) for v in y]

    x0 = [0.5, -0.5]
    lr = 0.25
    result = prox_method(f, grad_f, prox_g, x0, lr, n_iter=5, relaxation=1.3)
    assert isinstance(result, dict)
    assert "x" in result
    assert "objective" in result
    assert "n_iter" in result
    assert "lr" in result
    assert "relaxation" in result
    assert "step_norm" in result
    assert "method" in result
    assert isinstance(result["x"], list)
    assert len(result["x"]) == len(x0)
    assert result["n_iter"] == 5
    assert math.isclose(result["lr"], 0.25)
    assert math.isclose(result["relaxation"], 1.3)
    assert math.isfinite(result["objective"])
    assert math.isfinite(result["step_norm"])
