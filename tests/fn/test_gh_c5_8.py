"""Verification tests for gh_c5_8.

Ghosal and van der Vaart (2017), sec. 5.5, a Dirichlet-process mixture of Gaussian kernels.
"""

import math

import pytest

from morie.fn.gh_c5_8 import ghosal_gauss_ker


def test_the_mixture_is_a_density_on_the_grid():
    # Sec. 5.5: a DP mixture of Gaussian location-scale kernels
    x = [0.1, -0.3, 0.6, 0.2]
    res = ghosal_gauss_ker(x, K=12, alpha=1.0, seed=3)
    grid = [float(v) for v in res["grid"]]
    dens = [float(v) for v in res["density"]]
    step = grid[1] - grid[0]
    assert all(v >= 0.0 for v in dens)
    # the grid spans a bounded window, so the Gaussian tails outside it
    # are missing from the sum: the mass is just under one, never over
    total = sum(dens) * step
    assert 0.95 <= total <= 1.0 + 1e-9
    assert res["is_density"] is True


def test_the_stick_breaking_truncation_leaves_negligible_mass():
    x = [0.1, -0.3, 0.6, 0.2]
    res = ghosal_gauss_ker(x, K=30, alpha=1.0, seed=3)
    assert float(res["truncation_mass"]) < 1e-6
