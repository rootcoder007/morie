"""Verification tests for msm328.

The stub generator stamped several extracted page fragments of
Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, with the
same function name, so the implementation lives once in msm327 and
this module re-exports it. Its own contract is that both import paths
reach the same object; the value below is recomputed here as well, and
msm327's tests check the equation in full.
"""

import math

import pytest

import morie.fn.msm327 as host
import morie.fn.msm328 as alias
from morie.fn.msm328 import mvsml_functional_regression_eq_15_3


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert mvsml_functional_regression_eq_15_3 is getattr(host, "mvsml_functional_regression_eq_15_3")
    assert alias.mvsml_functional_regression_eq_15_3 is getattr(host, "mvsml_functional_regression_eq_15_3")


def test_every_exported_name_is_that_same_one_function():
    # the module also binds the equation-numbered name to it, so both
    # public names must resolve to the single implementation
    assert "mvsml_functional_regression_eq_15_3" in alias.__all__
    for name in alias.__all__:
        assert getattr(alias, name) is getattr(host, "mvsml_functional_regression_eq_15_3")


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm328" in alias.cheatsheet()


def test_the_prediction_through_the_alias_is_the_zero_altered_mean():
    # eq 15.3: (1 - theta) mu / (1 - exp(-mu))
    res = mvsml_functional_regression_eq_15_3(0.25, 2.0)
    assert res["estimate"] == pytest.approx(
        0.75 * 2.0 / (1.0 - math.exp(-2.0)), rel=1e-12)
