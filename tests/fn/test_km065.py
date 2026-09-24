"""Verification tests for km065.

Kamath, Keenan, Somers and Sorenson (2024), eq. 5.1, the pairwise reward-model loss. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km065 import kamath_ch5_reward_loss_pairwise


def _reward(x, y):
    """A fixed reward table, so the expected loss is computable by hand."""
    return {"good": 2.0, "bad": 0.5}[y]


def test_pairwise_reward_loss_is_the_mean_negative_log_sigmoid_margin():
    # Eq 5.1: loss = -E[log sigma(r(x, y_i) - r(x, y_{1-i}))]
    x = ["q1", "q2"]
    y0 = ["good", "bad"]
    y1 = ["bad", "good"]
    i = [0, 1]
    res = kamath_ch5_reward_loss_pairwise(_reward, x, y0, y1, i)
    margins = [1.5, 1.5]
    expected = sum(-math.log(1.0 / (1.0 + math.exp(-m))) for m in margins) / 2.0
    assert res["margins"] == pytest.approx(margins, rel=1e-12)
    assert res["estimate"] == pytest.approx(expected, rel=1e-12)


def test_the_index_names_which_candidate_was_preferred():
    # flipping i flips the sign of the margin
    forward = kamath_ch5_reward_loss_pairwise(_reward, ["q"], ["good"], ["bad"], [0])
    reversed_ = kamath_ch5_reward_loss_pairwise(_reward, ["q"], ["good"], ["bad"], [1])
    assert forward["margins"][0] == pytest.approx(-reversed_["margins"][0], rel=1e-12)
    assert reversed_["estimate"] > forward["estimate"]


def test_a_zero_margin_costs_log_two():
    flat = lambda x, y: 1.0
    res = kamath_ch5_reward_loss_pairwise(flat, ["q"], ["a"], ["b"], [0])
    assert res["estimate"] == pytest.approx(math.log(2.0), rel=1e-12)


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError):
        kamath_ch5_reward_loss_pairwise(_reward, ["q"], ["good", "bad"], ["bad"], [0])
