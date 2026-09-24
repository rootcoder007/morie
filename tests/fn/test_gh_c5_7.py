"""Verification tests for gh_c5_7.

Ghosal and van der Vaart (2017), sec. 5.4, Newton's predictive recursion.
"""

import math

import pytest

from morie.fn.gh_c5_7 import ghosal_pred_rec


def test_predictive_recursion_returns_a_mixing_density_and_a_mixture():
    # Sec. 5.4: one sweep updates f_i on a theta grid; the mixing
    # density must integrate to one over that grid
    x = [0.2, -0.4, 1.1, 0.5, -0.9, 0.3]
    res = ghosal_pred_rec(x)
    grid = [float(v) for v in res["theta_grid"]]
    f = [float(v) for v in res["f_mixing"]]
    step = grid[1] - grid[0]
    # a rectangle sum over a finite theta grid, so the mass closes to
    # the grid's own resolution rather than to machine precision
    assert sum(f) * step == pytest.approx(1.0, abs=1e-3)
    assert all(v >= 0.0 for v in f)
    assert res["single_pass"] is True


def test_the_recursion_depends_on_the_order_of_the_data():
    # the documented property: it is a single pass, so order matters
    x = [0.2, -0.4, 1.1, 0.5, -0.9, 0.3]
    a = ghosal_pred_rec(x)
    b = ghosal_pred_rec(list(reversed(x)))
    fa = [float(v) for v in a["f_mixing"]]
    fb = [float(v) for v in b["f_mixing"]]
    assert a["order_dependent"] is True
    assert max(abs(p - q) for p, q in zip(fa, fb)) > 0.0
