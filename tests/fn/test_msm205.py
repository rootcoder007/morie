"""Verification tests for msm205.

The stub generator stamped several extracted page fragments of
Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, with the
same function name, so the implementation lives once in msm204 and
this module re-exports it. Its own contract is that both import paths
reach the same object; the value below is recomputed here as well, and
msm204's tests check the equation in full.
"""

import math

import pytest

import morie.fn.msm204 as host
import morie.fn.msm205 as alias
from morie.fn.msm205 import mvsml_ridge_lasso_elastic_eq_9_30


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert mvsml_ridge_lasso_elastic_eq_9_30 is getattr(host, "mvsml_ridge_lasso_elastic_eq_9_30")
    assert alias.mvsml_ridge_lasso_elastic_eq_9_30 is getattr(host, "mvsml_ridge_lasso_elastic_eq_9_30")


def test_every_exported_name_is_that_same_one_function():
    # the module also binds the equation-numbered name to it, so both
    # public names must resolve to the single implementation
    assert "mvsml_ridge_lasso_elastic_eq_9_30" in alias.__all__
    for name in alias.__all__:
        assert getattr(alias, name) is getattr(host, "mvsml_ridge_lasso_elastic_eq_9_30")


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm205" in alias.cheatsheet()


def test_the_slackness_condition_through_the_alias_holds_on_the_margin():
    # eq 9.30: alpha_i [y_i f(x_i) - 1] = 0
    X = [[1.0, 1.0], [3.0, 3.0], [-1.0, -1.0]]
    res = mvsml_ridge_lasso_elastic_eq_9_30([0.5, 0.0, 0.0], X, [1, 1, -1], -1.0, [1.0, 1.0])
    assert list(res["margin_slack"]) == pytest.approx([0.0, 4.0, 2.0],
                                                       abs=1e-12)
    assert res["satisfied"] is True
