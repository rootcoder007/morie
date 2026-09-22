"""Tests for gh_ap_i2.ghosal_dudley_entropy."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_ap_i2 import ghosal_dudley_entropy


def _reference_estimate(sigma, a, n_grid):
    """Independent reimplementation of the quadrature from the docstring.

    J = (sigma/n_grid) * sum_{i=0}^{n_grid-1} sqrt(eps_i^{-a}),
    eps_i = (i + 0.5) * sigma / n_grid.
    """
    total = 0.0
    for i in range(n_grid):
        eps = (i + 0.5) * sigma / n_grid
        total += math.sqrt(eps ** (-a)) * sigma / n_grid
    return total


def test_gh_ap_i2_basic():
    """Test basic functionality with a documented call (keyword args)."""
    sigma = 1.0
    a = 1.0
    n_grid = 2000
    result = ghosal_dudley_entropy(sigma=sigma,
                                   entropy_exponent=a,
                                   n_grid=n_grid)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "finite" in result
    assert "method" in result
    estimate = float(result["estimate"])
    assert math.isfinite(estimate)
    expected = _reference_estimate(sigma, a, n_grid)
    assert math.isclose(estimate, expected, rel_tol=1e-12, abs_tol=0.0)
    assert bool(result["finite"]) == (a < 2.0)


def test_gh_ap_i2_edge():
    """Test edge cases: a < 2 -> finite; a >= 2 -> not finite."""
    # sigma passed positionally as in the documented signature.
    r_finite = ghosal_dudley_entropy(1.0, 1.0, n_grid=500)
    assert bool(r_finite["finite"]) is True
    assert math.isfinite(float(r_finite["estimate"]))

    r_infinite = ghosal_dudley_entropy(1.0, 2.0, n_grid=500)
    assert bool(r_infinite["finite"]) is False

    r_super = ghosal_dudley_entropy(1.0, 3.0, n_grid=500)
    assert bool(r_super["finite"]) is False
