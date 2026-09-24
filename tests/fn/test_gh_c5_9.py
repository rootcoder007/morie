"""Verification tests for gh_c5_9.

Ghosal and van der Vaart (2017), sec. 5.5, a Dirichlet-process mixture of Beta kernels.
"""

import math

import pytest

from morie.fn.gh_c5_9 import ghosal_beta_ker


def test_the_beta_mixture_puts_no_mass_outside_the_unit_interval():
    # Sec. 5.5: a Be(a theta, a (1 - theta)) kernel has support [0, 1]
    x = [0.2, 0.5, 0.7, 0.4]
    res = ghosal_beta_ker(x, alpha=1.0, precision=20.0, seed=3)
    assert float(res["mass_outside_support"]) == pytest.approx(0.0, abs=1e-12)
    assert tuple(res["support"]) == (0.0, 1.0)


def test_the_beta_mixture_is_a_density_on_its_support():
    x = [0.2, 0.5, 0.7, 0.4]
    res = ghosal_beta_ker(x, alpha=1.0, precision=20.0, seed=3)
    grid = [float(v) for v in res["grid"]]
    dens = [float(v) for v in res["density"]]
    step = grid[1] - grid[0]
    assert all(v >= 0.0 for v in dens)
    assert sum(dens) * step == pytest.approx(1.0, abs=1e-2)
