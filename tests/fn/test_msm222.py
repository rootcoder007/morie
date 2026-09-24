"""Verification tests for msm222.

The stub generator stamped several extracted page fragments of
Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, with the
same function name, so the implementation lives once in msm218 and
this module re-exports it. Its own contract is that both import paths
reach the same object; the value below is recomputed here as well, and
msm218's tests check the equation in full.
"""

import math

import pytest

import morie.fn.msm218 as host
import morie.fn.msm222 as alias
from morie.fn.msm222 import softsvm


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert softsvm is getattr(host, "softsvm")
    assert alias.softsvm is getattr(host, "softsvm")


def test_every_exported_name_is_that_same_one_function():
    # the module also binds the equation-numbered name to it, so both
    # public names must resolve to the single implementation
    assert "softsvm" in alias.__all__
    for name in alias.__all__:
        assert getattr(alias, name) is getattr(host, "softsvm")


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm222" in alias.cheatsheet()


def test_the_soft_margin_through_the_alias_is_the_reciprocal_norm():
    # eqs 9.34 and 9.35, with beta returned unnormalised
    res = softsvm([[1.0, 1.0], [-1.0, -1.0]], [1, -1], 1.0)
    nb = math.sqrt(sum(b * b for b in res["beta"]))
    assert res["margin"] == pytest.approx(1.0 / nb, rel=1e-12)
    assert res["n_violating"] == 0
