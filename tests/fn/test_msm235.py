"""Verification tests for msm235.

The stub generator stamped several extracted page fragments of
Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, with the
same function name, so the implementation lives once in msm234 and
this module re-exports it. Its own contract is that both import paths
reach the same object; the value below is recomputed here as well, and
msm234's tests check the equation in full.
"""

import math

import pytest

import morie.fn.msm234 as host
import morie.fn.msm235 as alias
from morie.fn.msm235 import ksvmdual


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert ksvmdual is getattr(host, "ksvmdual")
    assert alias.ksvmdual is getattr(host, "ksvmdual")


def test_every_exported_name_is_that_same_one_function():
    # the module also binds the equation-numbered name to it, so both
    # public names must resolve to the single implementation
    assert "ksvmdual" in alias.__all__
    for name in alias.__all__:
        assert getattr(alias, name) is getattr(host, "ksvmdual")


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm235" in alias.cheatsheet()


def test_the_kernel_dual_through_the_alias_reduces_to_the_linear_one():
    # eq 9.46 with a linear kernel is eq 9.44
    from morie.fn.msm231 import svmsdual
    X, y = [[1.0, 1.0], [-1.0, -1.0]], [1, -1]
    k = ksvmdual(X, y, 1.0, kernel="linear")
    assert k["objective"] == pytest.approx(svmsdual(X, y, 1.0)["objective"],
                                            rel=1e-8)
