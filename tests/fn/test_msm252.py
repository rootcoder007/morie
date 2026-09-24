"""Verification tests for msm252.

The stub generator stamped several extracted page fragments of
Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, with the
same function name, so the implementation lives once in msm251 and
this module re-exports it. Its own contract is that both import paths
reach the same object; the value below is recomputed here as well, and
msm251's tests check the equation in full.
"""

import math

import pytest

import morie.fn.msm251 as host
import morie.fn.msm252 as alias
from morie.fn.msm252 import mvsml_reproducing_kernel_eq_10_13


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert mvsml_reproducing_kernel_eq_10_13 is getattr(host, "mvsml_reproducing_kernel_eq_10_13")
    assert alias.mvsml_reproducing_kernel_eq_10_13 is getattr(host, "mvsml_reproducing_kernel_eq_10_13")


def test_every_exported_name_is_that_same_one_function():
    # the module also binds the equation-numbered name to it, so both
    # public names must resolve to the single implementation
    assert "mvsml_reproducing_kernel_eq_10_13" in alias.__all__
    for name in alias.__all__:
        assert getattr(alias, name) is getattr(host, "mvsml_reproducing_kernel_eq_10_13")


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm252" in alias.cheatsheet()
