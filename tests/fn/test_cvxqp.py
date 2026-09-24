"""Tests for cvxqp.boyd_quadratic_program."""

import pytest

from morie.fn import _array_core as np

from morie.fn.cvxqp import boyd_quadratic_program


def test_cvxqp_basic():
    """Test basic functionality with an inequality constraint."""
    P = [[2.0, 0.0], [0.0, 4.0]]
    q = [-2.0, -8.0]

    # Inequality constraint: x1 + x2 <= 1 (active at optimum)
    G = [[1.0, 1.0]]
    h = [1.0]

    result = boyd_quadratic_program(P, q, G=G, h=h)
    assert isinstance(result, dict)
    assert "x" in result
    assert "value" in result
    assert "psd" in result
    assert "converged" in result
    assert "lambda" in result
    assert "nu" in result
    # Feasibility: inequality constraint is satisfied
    assert result["x"][0] + result["x"][1] <= 1.0 + 1e-8


def test_cvxqp_edge():
    """Test edge cases: indefinite P is rejected."""
    # Indefinite P should raise ValueError as per docstring
    P = [[1.0, 0.0], [0.0, -1.0]]
    q = [0.0, 0.0]
    with pytest.raises(ValueError):
        boyd_quadratic_program(P, q)
