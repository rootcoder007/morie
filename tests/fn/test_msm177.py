"""Verification tests for msm177.

The stub generator stamped several extracted page fragments of
Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, with the
same function name, so the implementation lives once in msm173 and
this module re-exports it. Its own contract is that both import paths
reach the same object; the value below is recomputed here as well, and
msm173's tests check the equation in full.
"""

import math

import pytest

import morie.fn.msm173 as host
import morie.fn.msm177 as alias
from morie.fn.msm177 import mvsml_ridge_lasso_elastic_eq_9_5


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert mvsml_ridge_lasso_elastic_eq_9_5 is getattr(host, "mvsml_ridge_lasso_elastic_eq_9_5")
    assert alias.mvsml_ridge_lasso_elastic_eq_9_5 is getattr(host, "mvsml_ridge_lasso_elastic_eq_9_5")


def test_every_exported_name_is_that_same_one_function():
    # the module also binds the equation-numbered name to it, so both
    # public names must resolve to the single implementation
    assert "mvsml_ridge_lasso_elastic_eq_9_5" in alias.__all__
    for name in alias.__all__:
        assert getattr(alias, name) is getattr(host, "mvsml_ridge_lasso_elastic_eq_9_5")


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm177" in alias.cheatsheet()


def test_the_fitting_function_through_the_alias_is_the_inner_product():
    # eq 9.5: f(x_i) = beta_0 + x_i' beta
    res = mvsml_ridge_lasso_elastic_eq_9_5([[1.0, 2.0], [-1.0, -1.0]], 0.5, [1.0, 2.0])
    assert list(res["f"]) == pytest.approx([5.5, -2.5], rel=1e-12)
    assert list(res["labels"]) == [1, -1]
