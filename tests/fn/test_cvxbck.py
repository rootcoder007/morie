"""Tests for cvxbck.boyd_backtracking."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.cvxbck import boyd_backtracking


def test_cvxbck_basic():
    """Test basic functionality on a quadratic objective."""
    # f(x) = x @ x, grad = 2x; from x = [1.0] a full steepest-descent step overshoots
    x = np.array([1.0])
    f = lambda z: float(z @ z)
    grad = 2 * x
    dx = -grad  # descent direction
    result = boyd_backtracking(f, grad, x, dx)
    assert isinstance(result, dict)
    for key in ("t", "x_new", "f_new", "n_backtracks", "converged"):
        assert key in result
    assert math.isfinite(float(result["t"]))
    assert math.isfinite(float(result["f_new"]))
    assert result["t"] > 0
    assert result["f_new"] < f(x)


def test_cvxbck_edge():
    """Test that an ascent direction is rejected with ValueError."""
    x = np.array([1.0])
    f = lambda z: float(z @ z)
    grad = 2 * x
    dx = 2 * x  # ascent direction: grad @ dx > 0
    with pytest.raises(ValueError):
        boyd_backtracking(f, grad, x, dx)
