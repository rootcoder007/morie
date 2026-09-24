"""Verification tests for msm285.

The stub generator stamped several extracted page fragments of
Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, with the
same function name, so the implementation lives once in msm283 and
this module re-exports it. Its own contract is that both import paths
reach the same object; the value below is recomputed here as well, and
msm283's tests check the equation in full.
"""

import math

import pytest

import morie.fn.msm283 as host
import morie.fn.msm285 as alias
from morie.fn.msm285 import penfreg


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert penfreg is getattr(host, "penfreg")
    assert alias.penfreg is getattr(host, "penfreg")


def test_every_exported_name_is_that_same_one_function():
    # the module also binds the equation-numbered name to it, so both
    # public names must resolve to the single implementation
    assert "penfreg" in alias.__all__
    for name in alias.__all__:
        assert getattr(alias, name) is getattr(host, "penfreg")


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm285" in alias.cheatsheet()


def test_the_rotated_ridge_through_the_alias_is_least_squares_unpenalised():
    # eq 14.12 at lambda = 0, with X the identity and mu the mean
    res = penfreg([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]],
               [[1.0, 0.0], [0.0, 1.0]], 0.0)
    assert res["mu"] == pytest.approx(1.5, rel=1e-12)
    assert list(res["beta"]) == pytest.approx([-0.5, 0.5], rel=1e-12)
