"""Verification tests for km076.

Kamath, Keenan, Somers and Sorenson (2024), eq. 5.12, the direct preference optimisation loss. Expected values are
recomputed in the test body.
"""

import math
import statistics

import pytest

from morie.fn.km076 import kamath_ch5_dpo_loss


def test_dpo_loss_is_the_negative_log_sigmoid_of_the_beta_scaled_margin():
    # Eq 5.12: L = -E[log sigma(beta log(pi/pi_ref)_w - beta log(pi/pi_ref)_l)]
    theta = [(0.6, 0.2), (0.7, 0.1)]
    ref = [(0.4, 0.4), (0.5, 0.3)]
    beta = 0.1
    res = kamath_ch5_dpo_loss(theta, ref, beta)
    losses = []
    for (tw, tl), (rw, rl) in zip(theta, ref):
        margin = beta * math.log(tw / rw) - beta * math.log(tl / rl)
        losses.append(-math.log(1.0 / (1.0 + math.exp(-margin))))
    assert res["estimate"] == pytest.approx(sum(losses) / len(losses), rel=1e-12)


def test_the_implicit_reward_is_the_beta_scaled_log_ratio():
    theta = [(0.6, 0.2)]
    ref = [(0.4, 0.4)]
    beta = 0.1
    res = kamath_ch5_dpo_loss(theta, ref, beta)
    assert res["implicit_reward_w"][0] == pytest.approx(
        beta * math.log(0.6 / 0.4), rel=1e-12)
    assert res["implicit_reward_l"][0] == pytest.approx(
        beta * math.log(0.2 / 0.4), rel=1e-12)
    assert res["margins"][0] == pytest.approx(
        res["implicit_reward_w"][0] - res["implicit_reward_l"][0], rel=1e-12)


def test_a_policy_equal_to_the_reference_costs_log_two():
    # no preference is expressed, so the margin is zero
    res = kamath_ch5_dpo_loss([(0.3, 0.3)], [(0.3, 0.3)], 0.5)
    assert res["estimate"] == pytest.approx(math.log(2.0), rel=1e-12)


def test_moving_probability_onto_the_winner_lowers_the_loss():
    weak = kamath_ch5_dpo_loss([(0.35, 0.3)], [(0.3, 0.3)], 1.0)["estimate"]
    strong = kamath_ch5_dpo_loss([(0.9, 0.05)], [(0.3, 0.3)], 1.0)["estimate"]
    assert strong < weak
