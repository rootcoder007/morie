"""Verification tests for msm288.

The stub generator stamped several extracted page fragments of
Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, with the
same function name, so the implementation lives once in msm278 and
this module re-exports it. Its own contract is that both import paths
reach the same object; the value below is recomputed here as well, and
msm278's tests check the equation in full.
"""

import math

import pytest

import morie.fn.msm278 as host
import morie.fn.msm288 as alias
from morie.fn.msm288 import penmat


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert penmat is getattr(host, "penmat")
    assert alias.penmat is getattr(host, "penmat")


def test_every_exported_name_is_that_same_one_function():
    # the module also binds the equation-numbered name to it, so both
    # public names must resolve to the single implementation
    assert "penmat" in alias.__all__
    for name in alias.__all__:
        assert getattr(alias, name) is getattr(host, "penmat")


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm288" in alias.cheatsheet()


def test_the_roughness_matrix_through_the_alias_is_symmetric():
    # eq 14.11: P_ij is an integral symmetric in i and j
    res = penmat([0.0, 0.25, 0.5, 0.75, 1.0], 3, p=2)
    P = res["P"]
    for i in range(3):
        for j in range(3):
            assert P[i][j] == pytest.approx(P[j][i], rel=1e-9, abs=1e-12)
    assert res["order"] == 2
