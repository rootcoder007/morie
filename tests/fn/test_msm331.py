"""Verification tests for msm331.

The stub generator stamped several extracted page fragments of
Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, with the
same function name, so the implementation lives once in msm329 and
this module re-exports it. Its own contract is that both import paths
reach the same object; the value below is recomputed here as well, and
msm329's tests check the equation in full.
"""

import math

import pytest

import morie.fn.msm329 as host
import morie.fn.msm331 as alias
from morie.fn.msm331 import mvsml_functional_regression_eq_15_4


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert mvsml_functional_regression_eq_15_4 is getattr(host, "mvsml_functional_regression_eq_15_4")
    assert alias.mvsml_functional_regression_eq_15_4 is getattr(host, "mvsml_functional_regression_eq_15_4")


def test_every_exported_name_is_that_same_one_function():
    # the module also binds the equation-numbered name to it, so both
    # public names must resolve to the single implementation
    assert "mvsml_functional_regression_eq_15_4" in alias.__all__
    for name in alias.__all__:
        assert getattr(alias, name) is getattr(host, "mvsml_functional_regression_eq_15_4")


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm331" in alias.cheatsheet()


def test_the_threshold_rule_through_the_alias_switches_at_one_half():
    # eq 15.4: zero when theta > 0.5, the count mean otherwise
    assert mvsml_functional_regression_eq_15_4(0.25, 2.0)["estimate"] == pytest.approx(2.0, rel=1e-12)
    assert mvsml_functional_regression_eq_15_4(0.75, 2.0)["is_zero"] is True
