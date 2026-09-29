"""Tests for sqpmin.sequential_quadratic_programming."""

import pytest

from morie.fn.sqpmin import sequential_quadratic_programming


def test_equality_and_inequality_constrained_quadratic():
    """min (x-2)^2 + (y-1)^2 s.t. x + y = 2, x >= 0: Lagrange point (1.5, 0.5)
    with multiplier lambda = -1 on the equality (grad f = lambda grad c)."""
    r = sequential_quadratic_programming(
        lambda z: (z[0] - 2) ** 2 + (z[1] - 1) ** 2,
        [0.0, 0.0],
        eq=[lambda z: z[0] + z[1] - 2],
        ineq=[lambda z: z[0]],
    )
    assert r["converged"]
    assert r["x"] == pytest.approx([1.5, 0.5], abs=1e-7)
    assert r["multipliers_eq"] == pytest.approx([-1.0], abs=1e-6)
