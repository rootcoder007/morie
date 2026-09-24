"""Verification tests for msm172.

The stub generator stamped several extracted page fragments of
Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, with the
same function name, so the implementation lives once in msm161 and
this module re-exports it. Its own contract is that both import paths
reach the same object; the value below is recomputed here as well, and
msm161's tests check the equation in full.
"""

import math

import pytest

import morie.fn.msm161 as host
import morie.fn.msm172 as alias
from morie.fn.msm172 import hyperpl


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert hyperpl is getattr(host, "hyperpl")
    assert alias.hyperpl is getattr(host, "hyperpl")


def test_every_exported_name_is_that_same_one_function():
    # the module also binds the equation-numbered name to it, so both
    # public names must resolve to the single implementation
    assert "hyperpl" in alias.__all__
    for name in alias.__all__:
        assert getattr(alias, name) is getattr(host, "hyperpl")


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm172" in alias.cheatsheet()


def test_the_hyperplane_value_through_the_alias_is_the_affine_form():
    # eq 9.1: beta_0 + beta_1 x_1 + beta_2 x_2
    res = hyperpl([[1.0, 1.0], [0.0, 0.0]], -1.0, [1.0, 1.0])
    assert list(res["value"]) == pytest.approx([1.0, -1.0], abs=1e-12)
    assert res["norm_beta"] == pytest.approx(math.sqrt(2.0), rel=1e-12)
