"""Verification tests for msm136.

The stub generator stamped several extracted page fragments of
Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, with the
same function name, so the implementation lives once in msm135 and
this module re-exports it. Its own contract is that both import paths
reach the same object; the value below is recomputed here as well, and
msm135's tests check the equation in full.
"""

import math

import pytest

import morie.fn.msm135 as host
import morie.fn.msm136 as alias
from morie.fn.msm136 import mvsml_categorical_count_eq_8_6


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert mvsml_categorical_count_eq_8_6 is getattr(host, "mvsml_categorical_count_eq_8_6")
    assert alias.mvsml_categorical_count_eq_8_6 is getattr(host, "mvsml_categorical_count_eq_8_6")


def test_every_exported_name_is_that_same_one_function():
    # the module also binds the equation-numbered name to it, so both
    # public names must resolve to the single implementation
    assert "mvsml_categorical_count_eq_8_6" in alias.__all__
    for name in alias.__all__:
        assert getattr(alias, name) is getattr(host, "mvsml_categorical_count_eq_8_6")


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm136" in alias.cheatsheet()


def test_the_estimating_equations_through_the_alias_stay_consistent():
    # eq 8.6, second block: K'C theta + (K'K + lambda K sigma2_e) beta = K'y
    C, K, y = [[1.0], [1.0]], [[2.0, 0.0], [0.0, 3.0]], [1.0, 3.0]
    res = mvsml_categorical_count_eq_8_6(C, K, y, lam=1.0, sigma2_e=1.0)
    th, be = list(res["theta"]), list(res["beta"])
    for i in range(2):
        lhs = sum(K[r][i] * C[r][0] * th[0] for r in range(2))
        lhs += sum((sum(K[r][i] * K[r][j] for r in range(2))
                    + 1.0 * K[i][j]) * be[j] for j in range(2))
        assert lhs == pytest.approx(sum(K[r][i] * y[r] for r in range(2)),
                                     rel=1e-9)
