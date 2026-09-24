"""Verification tests for msm185.

The stub generator stamped several extracted page fragments of
Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, with the
same function name, so the implementation lives once in msm184 and
this module re-exports it. Its own contract is that both import paths
reach the same object; the value below is recomputed here as well, and
msm184's tests check the equation in full.
"""

import math

import pytest

import morie.fn.msm184 as host
import morie.fn.msm185 as alias
from morie.fn.msm185 import wolfedual


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert wolfedual is getattr(host, "wolfedual")
    assert alias.wolfedual is getattr(host, "wolfedual")


def test_every_exported_name_is_that_same_one_function():
    # the module also binds the equation-numbered name to it, so both
    # public names must resolve to the single implementation
    assert "wolfedual" in alias.__all__
    for name in alias.__all__:
        assert getattr(alias, name) is getattr(host, "wolfedual")


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm185" in alias.cheatsheet()


def test_the_wolfe_dual_through_the_alias_subtracts_the_multipliers():
    # eq 9.12: L = f - sum lambda_i h_i - sum alpha_i g_i
    res = wolfedual(10.0, [1.0], h=[2.0], grad_h=[[1.0]], g=[3.0],
               grad_g=[[1.0]], lam=[0.5], alpha=[2.0])
    assert res["L"] == pytest.approx(10.0 - 0.5 * 2.0 - 2.0 * 3.0,
                                      rel=1e-12)
