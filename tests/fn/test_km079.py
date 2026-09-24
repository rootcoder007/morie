"""Verification tests for km079.

Kamath, Keenan, Somers and Sorenson (2024), eq. 6.3, the AlignScore joint loss. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km079 import kamath_ch6_alignscore_total_loss


def test_alignscore_total_is_the_weighted_sum_of_the_three_heads():
    # Eq 6.3: L = lam1 L_3way + lam2 L_bin + lam3 L_reg
    losses = (1.0, 2.0, 3.0)
    lambdas = [0.2, 0.3, 0.5]
    res = kamath_ch6_alignscore_total_loss(*losses, lambdas)
    assert res["estimate"] == pytest.approx(
        sum(l * w for l, w in zip(losses, lambdas)), rel=1e-12)
    for got, (l, w) in zip(res["contributions"], zip(losses, lambdas)):
        assert got == pytest.approx(l * w, rel=1e-12)


def test_all_weight_on_one_head_returns_that_loss():
    res = kamath_ch6_alignscore_total_loss(1.0, 2.0, 3.0, [0.0, 1.0, 0.0])
    assert res["estimate"] == pytest.approx(2.0, rel=1e-12)


def test_equal_weights_give_the_mean_of_the_heads():
    third = 1.0 / 3.0
    res = kamath_ch6_alignscore_total_loss(1.0, 2.0, 3.0, [third, third, third])
    assert res["estimate"] == pytest.approx(2.0, rel=1e-12)
