"""Verification tests for msm225.

The stub generator stamped several extracted page fragments of
Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, with the
same function name, so the implementation lives once in msm223 and
this module re-exports it. Its own contract is that both import paths
reach the same object; the value below is recomputed here as well, and
msm223's tests check the equation in full.
"""

import math

import pytest

import morie.fn.msm223 as host
import morie.fn.msm225 as alias
from morie.fn.msm225 import svmkkt


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert svmkkt is getattr(host, "svmkkt")
    assert alias.svmkkt is getattr(host, "svmkkt")


def test_every_exported_name_is_that_same_one_function():
    # the module also binds the equation-numbered name to it, so both
    # public names must resolve to the single implementation
    assert "svmkkt" in alias.__all__
    for name in alias.__all__:
        assert getattr(alias, name) is getattr(host, "svmkkt")


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm225" in alias.cheatsheet()


def test_the_wolfe_primal_through_the_alias_reaches_its_optimum():
    # eq 9.38 with every bracket of 9.42 and 9.43 vanishing
    X, y = [[1.0, 1.0], [-1.0, -1.0]], [1, -1]
    res = svmkkt(X, y, 0.0, [0.5, 0.5], [0.25, 0.25], [0.75, 0.75],
               [0.0, 0.0], 1.0)
    assert res["L"] == pytest.approx(0.5 * (0.25 + 0.25), rel=1e-12)
    assert res["balance"] == pytest.approx(0.0, abs=1e-12)
    assert res["kkt_satisfied"] is True
