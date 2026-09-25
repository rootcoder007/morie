"""Tests for wsmexp.wasserman_expectation."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.wsmexp import wasserman_expectation


def test_wsmexp_basic():
    """Uniform(a, b) has mean (a + b) / 2 and unit mass."""
    a, b = 0.0, 1.0
    n = 20001
    g = np.linspace(a, b, n)
    dens = [1.0 / (b - a)] * n
    out = wasserman_expectation(g, dens)
    assert out["n"] == n
    assert out["estimate"] == pytest.approx(0.5 * (a + b), abs=1e-12)
    assert out["density_mass"] == pytest.approx(1.0, abs=1e-12)

    # Shifting the support shifts the mean by the same amount.
    shifted = wasserman_expectation(np.linspace(a + 7.0, b + 7.0, n), dens)
    assert shifted["estimate"] == pytest.approx(out["estimate"] + 7.0, rel=1e-12)
    assert shifted["density_mass"] == pytest.approx(1.0, abs=1e-12)


def test_wsmexp_triangular_and_exponential_means():
    """Known closed-form means on their own grids."""
    # Triangular density f(x) = 2x on [0, 1] has mean 2/3.
    n = 40001
    g = np.linspace(0.0, 1.0, n)
    tri = wasserman_expectation(g, 2.0 * g)
    assert tri["density_mass"] == pytest.approx(1.0, abs=1e-9)
    assert tri["estimate"] == pytest.approx(2.0 / 3.0, abs=1e-9)

    # Exponential(rate) truncated far out in the tail has mean 1/rate.
    rate = 2.0
    gx = np.linspace(0.0, 40.0 / rate, 40001)
    expo = wasserman_expectation(gx, rate * np.exp(-rate * gx))
    assert expo["density_mass"] == pytest.approx(1.0, abs=1e-6)
    assert expo["estimate"] == pytest.approx(1.0 / rate, abs=1e-6)


def test_wsmexp_does_not_renormalise():
    """A density integrating to 2 is reported, not silently rescaled."""
    n = 10001
    g = np.linspace(0.0, 1.0, n)
    out = wasserman_expectation(g, [2.0] * n)
    assert out["density_mass"] == pytest.approx(2.0, abs=1e-12)
    # int x * 2 dx over [0, 1] = 1, i.e. twice the uniform mean.
    assert out["estimate"] == pytest.approx(1.0, abs=1e-12)


def test_wsmexp_edge():
    """Short grids, length mismatch, non-monotone grids and negative
    densities are all rejected; a two-point grid is exact."""
    g = np.linspace(0.0, 1.0, 101)
    with pytest.raises(ValueError):
        wasserman_expectation([0.0], [1.0])
    with pytest.raises(ValueError):
        wasserman_expectation([0.0, 1.0, 2.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        wasserman_expectation([1.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        wasserman_expectation([0.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        wasserman_expectation(g, [-1.0] * g.size)

    # Trapezoid on two points: int_0^2 x*(f0 + (f1-f0)x/2)/... check the
    # rule directly with f == 1, where E[X] = 0.5 * dx * (x0 + x1) = 2.
    two = wasserman_expectation([0.0, 4.0], [1.0, 1.0])
    assert two["n"] == 2
    assert two["density_mass"] == pytest.approx(4.0, abs=1e-12)
    assert two["estimate"] == pytest.approx(
        0.5 * (4.0 - 0.0) * (0.0 * 1.0 + 4.0 * 1.0), rel=1e-12
    )

    # A zero density gives zero mass and zero expectation, without error.
    zero = wasserman_expectation([0.0, 1.0], [0.0, 0.0])
    assert zero["density_mass"] == 0.0
    assert zero["estimate"] == 0.0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmexp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
