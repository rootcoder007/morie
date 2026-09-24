"""Verification tests for msm184.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 9, eqs. 9.9 to 9.14 pp.346-347, the Wolfe dual. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm184 import wolfedual


def test_the_dual_objective_subtracts_both_multiplier_terms():
    # eq 9.12: L = f - sum lambda_i h_i - sum alpha_i g_i
    res = wolfedual(10.0, [1.0], h=[2.0], grad_h=[[1.0]], g=[3.0],
               grad_g=[[1.0]], lam=[0.5], alpha=[2.0])
    assert res["L"] == pytest.approx(
        10.0 - 0.5 * 2.0 - 2.0 * 3.0, rel=1e-12)


def test_the_book_s_first_illustrative_example_is_stationary_at_its_optimum():
    # minimize x^2 subject to x >= 1, whose dual the book writes as
    # x^2 - 2 alpha (x - 1); at x = 1 the general form needs alpha = 2
    res = wolfedual(1.0, [2.0], g=[0.0], grad_g=[[1.0]], alpha=[2.0])
    assert res["L"] == pytest.approx(1.0, rel=1e-12)
    assert list(res["stationarity"]) == pytest.approx([0.0], abs=1e-12)
    assert res["alpha_nonnegative"] is True
    assert res["max_stationarity"] == pytest.approx(0.0, abs=1e-12)


def test_a_negative_multiplier_breaks_the_inequality_condition():
    # eq 9.14 requires alpha_i >= 0
    res = wolfedual(1.0, [2.0], g=[0.0], grad_g=[[1.0]], alpha=[-2.0])
    assert res["alpha_nonnegative"] is False


def test_an_off_optimum_point_leaves_a_stationarity_residual():
    res = wolfedual(4.0, [4.0], g=[1.0], grad_g=[[1.0]], alpha=[2.0])
    assert abs(res["stationarity"][0]) > 1e-9
