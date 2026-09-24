"""Verification tests for gh_c9_2.

Ghosal and van der Vaart (2017), sec. 9.2, contraction of the Dirichlet-process posterior CDF.
"""

import math

import pytest

from morie.fn.gh_c9_2 import ghosal_dp_disc_crt


def test_the_dirichlet_posterior_mean_contracts_at_the_root_n_rate():
    # sec. 9.2: sup-error is O_P(n^{-1/2}), so multiplying n by sixteen
    # roughly halves it twice
    res = ghosal_dp_disc_crt(ns=(100, 400, 1600), alpha=2.0, seed=9)
    errs = [float(v) for v in res["sup_err_by_n"]]
    assert errs[0] > errs[1] > errs[2]
    ratio = errs[0] / errs[2]
    assert 2.0 < ratio < 8.0


def test_the_posterior_mean_cdf_stays_a_probability():
    res = ghosal_dp_disc_crt(ns=(100, 400), alpha=2.0, seed=9)
    assert all(0.0 <= float(v) <= 1.0 for v in res["sup_err_by_n"])
