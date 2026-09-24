"""Verification tests for gh_c14_18.

Ghosal and van der Vaart (2017), sec. 14.9.1, kernel stick-breaking.
"""

import math

import pytest

from morie.fn.gh_c14_18 import ghosal_ksbp_def


def test_the_kernel_sticks_form_a_probability_distribution_at_each_covariate():
    # sec. 14.9.1: p_k(x) = V_k(x) prod_{j<k} (1 - V_j(x)) sums to one
    res = ghosal_ksbp_def(x=(0.2, 0.8), n_terms=60, seed=4)
    assert res["mass_x0"] == pytest.approx(1.0, abs=1e-6)
    assert res["mass_x1"] == pytest.approx(1.0, abs=1e-6)


def test_the_weights_depend_on_the_covariate():
    res = ghosal_ksbp_def(x=(0.2, 0.8), n_terms=60, seed=4)
    assert res["weights_vary_with_x"] is True
