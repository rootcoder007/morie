"""Verification tests for msm099.

The stub generator stamped several extracted page fragments of
Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, with the
same function name, so the implementation lives once in msm092 and
this module re-exports it. Its own contract is that both import paths
reach the same object; the value below is recomputed here as well, and
msm092's tests check the equation in full.
"""

import math

import pytest

import morie.fn.msm092 as host
import morie.fn.msm099 as alias
from morie.fn.msm099 import mvsml_bayesian_regression_pt2_eq_7_3


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert mvsml_bayesian_regression_pt2_eq_7_3 is getattr(host, "mvsml_bayesian_regression_pt2_eq_7_3")
    assert alias.mvsml_bayesian_regression_pt2_eq_7_3 is getattr(host, "mvsml_bayesian_regression_pt2_eq_7_3")


def test_every_exported_name_is_that_same_one_function():
    # the module also binds the equation-numbered name to it, so both
    # public names must resolve to the single implementation
    assert "mvsml_bayesian_regression_pt2_eq_7_3" in alias.__all__
    for name in alias.__all__:
        assert getattr(alias, name) is getattr(host, "mvsml_bayesian_regression_pt2_eq_7_3")


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm099" in alias.cheatsheet()


def test_the_latent_predictor_through_the_alias_counts_every_column():
    # eq 7.3: L = X_E beta_E + X beta + X_EM beta_EM + eps
    res = mvsml_bayesian_regression_pt2_eq_7_3(2, X_E=[[1.0, 0.0], [0.0, 1.0]],
               X=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert res["estimate"] == pytest.approx(2 + 3, rel=1e-12)
    assert res["widths"] == {"environments": 2, "markers": 3}
