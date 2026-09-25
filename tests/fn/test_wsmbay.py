"""Tests for wsmbay.wasserman_posterior."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.wsmbay import wasserman_posterior


def _trapz(y, x):
    """Trapezoid integral of the returned lists, computed here so the
    expected values do not depend on the module's own quadrature."""
    return sum(
        0.5 * (x[i + 1] - x[i]) * (y[i + 1] + y[i]) for i in range(len(x) - 1)
    )


def test_wsmbay_basic():
    """N(theta, 1) with a flat prior: posterior is N(xbar, 1/n)."""
    data = [1.0, 0.5, 1.5]
    n = len(data)
    xbar = sum(data) / n
    grid = np.linspace(-5.0, 5.0, 4001)
    out = wasserman_posterior(data, None, (grid, np.ones_like(grid)))

    assert out["n"] == n
    assert out["estimate"] == pytest.approx(xbar, abs=1e-9)
    assert abs(out["map_theta"] - xbar) < 0.005
    assert out["theta_grid"][0] == -5.0
    assert out["theta_grid"][-1] == 5.0
    assert len(out["posterior"]) == len(out["theta_grid"]) == 4001
    assert all(p >= 0.0 for p in out["posterior"])

    g, p = out["theta_grid"], out["posterior"]
    # A density integrates to 1 ...
    assert _trapz(p, g) == pytest.approx(1.0, abs=1e-12)
    # ... and the posterior variance of N(xbar, 1/n) is 1/n.
    var = _trapz([(t - xbar) ** 2 * pi for t, pi in zip(g, p)], g)
    assert var == pytest.approx(1.0 / n, abs=1e-6)

    # Its mode and mean coincide, and both sit at the sample mean.
    assert out["map_theta"] == pytest.approx(
        g[max(range(len(p)), key=lambda i: p[i])], rel=1e-12
    )


def test_wsmbay_conjugate_normal_prior_shrinks_the_mean():
    """N(0, 1) prior with the N(theta, 1) model: mean = n xbar / (n + 1)."""
    data = [1.0, 0.5, 1.5]
    n = len(data)
    xbar = sum(data) / n
    grid = np.linspace(-6.0, 6.0, 6001)
    prior = np.exp(-0.5 * grid ** 2)
    out = wasserman_posterior(data, None, (grid, prior))

    assert out["estimate"] == pytest.approx(n * xbar / (n + 1.0), abs=1e-6)
    # Posterior variance is 1/(n + 1) for this conjugate pair.
    g, p = out["theta_grid"], out["posterior"]
    mean = out["estimate"]
    var = _trapz([(t - mean) ** 2 * pi for t, pi in zip(g, p)], g)
    assert var == pytest.approx(1.0 / (n + 1.0), abs=1e-6)
    # Shrinkage pulls the posterior mean strictly below the sample mean.
    assert out["estimate"] < xbar


def test_wsmbay_custom_density_matches_the_default():
    """Passing the N(theta, 1) density explicitly reproduces f = None."""
    data = [0.2, -0.4, 1.1, 0.7]
    grid = np.linspace(-5.0, 5.0, 2001)
    flat = np.ones_like(grid)

    def normal(x, th):
        return np.exp(-0.5 * (x - th) ** 2) / math.sqrt(2.0 * math.pi)

    a = wasserman_posterior(data, None, (grid, flat))
    b = wasserman_posterior(data, normal, (grid, flat))
    assert b["estimate"] == pytest.approx(a["estimate"], rel=1e-12)
    assert b["map_theta"] == pytest.approx(a["map_theta"], rel=1e-12)
    assert b["evidence"] == pytest.approx(a["evidence"], rel=1e-12)
    # Flat prior: the evidence is the likelihood integrated over the grid.
    assert a["evidence"] > 0.0


def test_wsmbay_edge():
    """Empty data and malformed priors are rejected."""
    grid = np.linspace(-5.0, 5.0, 501)
    flat = np.ones_like(grid)

    with pytest.raises(ValueError):
        wasserman_posterior([], None, (grid, flat))
    # Grid must be strictly increasing.
    with pytest.raises(ValueError):
        wasserman_posterior([0.0], None, ([1.0, 0.0], [1.0, 1.0]))
    # Prior density cannot be negative.
    with pytest.raises(ValueError):
        wasserman_posterior([0.0], None, (grid, [-1.0] * grid.size))
    # Fewer than two grid points.
    with pytest.raises(ValueError):
        wasserman_posterior([0.0], None, ([0.0], [1.0]))
    # Mismatched grid and density lengths.
    with pytest.raises(ValueError):
        wasserman_posterior([0.0], None, ([0.0, 1.0, 2.0], [1.0, 1.0]))
    # A prior that puts no mass anywhere the likelihood lives.
    with pytest.raises(ValueError):
        wasserman_posterior([0.0], None, (grid, [0.0] * grid.size))

    # A single observation with a flat prior centres the posterior on it.
    one = wasserman_posterior([2.0], None, (np.linspace(-8.0, 12.0, 4001), [1.0] * 4001))
    assert one["n"] == 1
    assert one["estimate"] == pytest.approx(2.0, abs=1e-9)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmbay as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
