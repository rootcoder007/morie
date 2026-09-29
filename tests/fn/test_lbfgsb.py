"""Tests for lbfgsb.lbfgsb_minimize (Byrd, Lu, Nocedal and Zhu 1995)."""

import pytest

from morie.fn.lbfgsb import lbfgsb_minimize


def test_box_constrained_quadratic_lands_on_the_projected_minimiser():
    """min (x - 2)^2 + (y + 1)^2 on [0, 1] x [0, 3] is the projection (1, 0)
    of the unconstrained minimiser, where the projected gradient vanishes."""

    def f(z):
        return (z[0] - 2.0) ** 2 + (z[1] + 1.0) ** 2

    def g(z):
        return [2 * (z[0] - 2.0), 2 * (z[1] + 1.0)]

    r = lbfgsb_minimize(f, [0.5, 2.0], grad=g, lower=[0.0, 0.0], upper=[1.0, 3.0])
    assert r["x"] == pytest.approx([1.0, 0.0], abs=1e-10)
    assert r["fun"] == pytest.approx(2.0, abs=1e-10)
    assert r["projected_gradient"] <= 1e-8


def test_unconstrained_quadratic():
    r = lbfgsb_minimize(lambda z: (z[0] - 3.0) ** 2 + 2 * (z[1] - 1.0) ** 2, [0.0, 0.0])
    assert r["x"] == pytest.approx([3.0, 1.0], abs=1e-6)
