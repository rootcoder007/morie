"""Tests for wsmbcr.wasserman_credible_interval."""

import pytest

from morie.fn import _array_core as np
from morie.fn.wsmbay import wasserman_posterior
from morie.fn.wsmbcr import wasserman_credible_interval


def test_wsmbcr_basic():
    """Uniform posterior on [0, 1]: the interval is (alpha/2, 1 - alpha/2)."""
    alpha = 0.10
    g = np.linspace(0.0, 1.0, 20001)
    out = wasserman_credible_interval((g, np.ones_like(g)), alpha)

    assert out["alpha"] == alpha
    assert out["lower"] == pytest.approx(alpha / 2.0, abs=1e-9)
    assert out["upper"] == pytest.approx(1.0 - alpha / 2.0, abs=1e-9)
    assert out["estimate"] == pytest.approx(1.0 - alpha, abs=1e-9)
    assert out["estimate"] == pytest.approx(out["upper"] - out["lower"], rel=1e-12)
    # A density of 1 on a unit interval already carries mass 1.
    assert out["mass_drift"] == pytest.approx(0.0, abs=1e-12)

    # Narrower level, wider interval; the tails stay equal.
    tight = wasserman_credible_interval((g, np.ones_like(g)), 0.5)
    assert tight["lower"] == pytest.approx(0.25, abs=1e-9)
    assert tight["upper"] == pytest.approx(0.75, abs=1e-9)
    assert tight["estimate"] < out["estimate"]


def test_wsmbcr_renormalises_an_unnormalised_density():
    """A density integrating to 2 reports drift 1 and the same interval."""
    alpha = 0.10
    g = np.linspace(0.0, 1.0, 20001)
    out = wasserman_credible_interval((g, [2.0] * g.size), alpha)
    assert out["mass_drift"] == pytest.approx(1.0, abs=1e-12)
    assert out["lower"] == pytest.approx(alpha / 2.0, abs=1e-9)
    assert out["upper"] == pytest.approx(1.0 - alpha / 2.0, abs=1e-9)


def test_wsmbcr_on_a_grid_posterior_from_wsmbay():
    """The N(theta, 1) flat-prior posterior is N(xbar, 1/n): the 95 percent
    equal-tail interval is xbar +/- 1.959963984540054 / sqrt(n)."""
    data = [1.0, 0.5, 1.5, 1.2, 0.8]
    n = len(data)
    xbar = sum(data) / n
    grid = np.linspace(-5.0, 5.0, 8001)
    post = wasserman_posterior(data, None, (grid, np.ones_like(grid)))

    alpha = 0.05
    out = wasserman_credible_interval((post["theta_grid"], post["posterior"]), alpha)
    half = 1.959963984540054 / n ** 0.5
    assert out["lower"] == pytest.approx(xbar - half, abs=1e-4)
    assert out["upper"] == pytest.approx(xbar + half, abs=1e-4)
    assert out["mass_drift"] == pytest.approx(0.0, abs=1e-9)
    # Centred on the posterior mean.
    assert 0.5 * (out["lower"] + out["upper"]) == pytest.approx(xbar, abs=1e-4)


def test_wsmbcr_edge():
    """alpha outside (0, 1), bad grids, and a zero-mass posterior are errors."""
    g = np.linspace(0.0, 1.0, 1001)
    flat = np.ones_like(g)

    for bad_alpha in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            wasserman_credible_interval((g, flat), bad_alpha)
    # Mismatched lengths.
    with pytest.raises(ValueError):
        wasserman_credible_interval(([0.0, 1.0, 2.0], [1.0, 1.0]), 0.1)
    # Fewer than two points.
    with pytest.raises(ValueError):
        wasserman_credible_interval(([0.0], [1.0]), 0.1)
    # Zero mass.
    with pytest.raises(ValueError):
        wasserman_credible_interval(([0.0, 1.0], [0.0, 0.0]), 0.1)

    # Two-point grid: the CDF is linear, so inversion is exact.
    two = wasserman_credible_interval(([0.0, 1.0], [1.0, 1.0]), 0.2)
    assert two["lower"] == pytest.approx(0.1, abs=1e-12)
    assert two["upper"] == pytest.approx(0.9, abs=1e-12)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmbcr as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
