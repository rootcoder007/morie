"""Verification tests for msm192.

The stub generator stamped several extracted page fragments of
Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, with the
same function name, so the implementation lives once in msm188 and
this module re-exports it. Its own contract is that both import paths
reach the same object; the value below is recomputed here as well, and
msm188's tests check the equation in full.
"""

import math

import pytest

import morie.fn.msm188 as host
import morie.fn.msm192 as alias
from morie.fn.msm192 import qplincon


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert qplincon is getattr(host, "qplincon")
    assert alias.qplincon is getattr(host, "qplincon")


def test_every_exported_name_is_that_same_one_function():
    # the module also binds the equation-numbered name to it, so both
    # public names must resolve to the single implementation
    assert "qplincon" in alias.__all__
    for name in alias.__all__:
        assert getattr(alias, name) is getattr(host, "qplincon")


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm192" in alias.cheatsheet()


def test_the_quadratic_program_through_the_alias_matches_the_book():
    # eq 9.20: alpha = c / (a'a), and Illustrative Example 9.2 prints
    # x = y = alpha = 1 for a = [1, 1], c = 2
    res = qplincon([1.0, 1.0], 2.0)
    assert res["alpha"] == pytest.approx(1.0, rel=1e-12)
    assert list(res["x"]) == pytest.approx([1.0, 1.0], rel=1e-12)
    assert res["primal_value"] == pytest.approx(res["dual_value"],
                                                 rel=1e-12)
