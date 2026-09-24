"""Verification tests for km067.

Kamath, Keenan, Somers and Sorenson (2024), eq. 5.3, the Bradley-Terry reward-model loss. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km067 import kamath_ch5_rm_bradley_terry


def _reward(x, y):
    """A fixed reward table, so the expected loss is computable by hand."""
    return {"good": 2.0, "bad": 0.5}[y]


def test_bradley_terry_loss_with_the_winner_already_named():
    # Eq 5.3: L = -E[log sigma(r(x, y_w) - r(x, y_l))]
    x = ["q1", "q2"]
    yw = ["good", "good"]
    yl = ["bad", "bad"]
    res = kamath_ch5_rm_bradley_terry(x, yw, yl, _reward)
    margin = 1.5
    expected = -math.log(1.0 / (1.0 + math.exp(-margin)))
    assert res["estimate"] == pytest.approx(expected, rel=1e-12)
    assert all(m == pytest.approx(margin, rel=1e-12) for m in res["margins"])


def test_a_wider_margin_costs_less():
    wide = lambda x, y: {"w": 5.0, "l": 0.0}[y]
    narrow = lambda x, y: {"w": 1.0, "l": 0.9}[y]
    a = kamath_ch5_rm_bradley_terry(["q"], ["w"], ["l"], wide)["estimate"]
    b = kamath_ch5_rm_bradley_terry(["q"], ["w"], ["l"], narrow)["estimate"]
    assert a < b


def test_the_loss_is_never_negative():
    # -log sigma(.) is non-negative because sigma <= 1
    res = kamath_ch5_rm_bradley_terry(["q"], ["good"], ["bad"], _reward)
    assert res["estimate"] >= 0.0
