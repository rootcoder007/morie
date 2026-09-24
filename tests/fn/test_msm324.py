"""Verification tests for msm324.

The stub generator stamped several extracted page fragments of
Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, with the
same function name, so the implementation lives once in msm323 and
this module re-exports it. Its own contract is that both import paths
reach the same object; the value below is recomputed here as well, and
msm323's tests check the equation in full.
"""

import math

import pytest

import morie.fn.msm323 as host
import morie.fn.msm324 as alias
from morie.fn.msm324 import mvsml_functional_regression_eq_15_1


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert mvsml_functional_regression_eq_15_1 is getattr(host, "mvsml_functional_regression_eq_15_1")
    assert alias.mvsml_functional_regression_eq_15_1 is getattr(host, "mvsml_functional_regression_eq_15_1")


def test_every_exported_name_is_that_same_one_function():
    # the module also binds the equation-numbered name to it, so both
    # public names must resolve to the single implementation
    assert "mvsml_functional_regression_eq_15_1" in alias.__all__
    for name in alias.__all__:
        assert getattr(alias, name) is getattr(host, "mvsml_functional_regression_eq_15_1")


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm324" in alias.cheatsheet()


def test_the_links_through_the_alias_invert_the_log_and_the_logit():
    # eq 15.1: log(mu) = f_mu and log(theta / (1 - theta)) = f_theta
    res = mvsml_functional_regression_eq_15_1(2.0, 0.25)
    assert res["mu"] == pytest.approx(math.exp(2.0), rel=1e-12)
    assert res["theta"] == pytest.approx(1.0 / (1.0 + math.exp(-0.25)),
                                          rel=1e-12)
