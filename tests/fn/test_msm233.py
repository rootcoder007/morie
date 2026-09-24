"""Verification tests for msm233.

The stub generator stamped several extracted page fragments of
Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, with the
same function name, so the implementation lives once in msm231 and
this module re-exports it. Its own contract is that both import paths
reach the same object; the value below is recomputed here as well, and
msm231's tests check the equation in full.
"""

import math

import pytest

import morie.fn.msm231 as host
import morie.fn.msm233 as alias
from morie.fn.msm233 import svmsdual


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert svmsdual is getattr(host, "svmsdual")
    assert alias.svmsdual is getattr(host, "svmsdual")


def test_every_exported_name_is_that_same_one_function():
    # the module also binds the equation-numbered name to it, so both
    # public names must resolve to the single implementation
    assert "svmsdual" in alias.__all__
    for name in alias.__all__:
        assert getattr(alias, name) is getattr(host, "svmsdual")


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm233" in alias.cheatsheet()


def test_the_soft_margin_dual_through_the_alias_finds_the_optimum():
    # eq 9.44 is 2a - 4a^2 for this pair, maximised at a = 1/4
    res = svmsdual([[1.0, 1.0], [-1.0, -1.0]], [1, -1], 1.0)
    assert list(res["alpha"]) == pytest.approx([0.25, 0.25], rel=1e-6)
    assert res["objective"] == pytest.approx(0.25, rel=1e-6)
    assert res["balance"] == pytest.approx(0.0, abs=1e-9)
