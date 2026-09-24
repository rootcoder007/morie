"""Verification tests for km066.

Kamath, Keenan, Somers and Sorenson (2024), ch 5, the KL-penalised reward, eq. 5.2. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km066 import kamath_ch5_reward_kl_penalty


def test_the_total_reward_subtracts_the_scaled_log_policy_ratio():
    # r_total = r_theta(x, y) - beta log(pi_RL(y|x) / pi_SFT(y|x))
    res = kamath_ch5_reward_kl_penalty("p", "resp", 0.5, 0.25, 2.0, r_theta=1.0)
    expected = 1.0 - 2.0 * math.log(0.5 / 0.25)
    assert res["estimate"] == pytest.approx(expected, rel=1e-12)
    assert res["estimate"] == pytest.approx(1.0 - 2.0 * math.log(2.0),
                                            rel=1e-12)


def test_a_zero_coefficient_leaves_the_bare_reward():
    res = kamath_ch5_reward_kl_penalty("p", "resp", 0.5, 0.25, 0.0, r_theta=1.0)
    assert res["estimate"] == pytest.approx(1.0, rel=1e-12)


def test_matching_the_reference_policy_costs_no_penalty():
    res = kamath_ch5_reward_kl_penalty("p", "resp", 0.25, 0.25, 3.0, r_theta=1.0)
    assert res["estimate"] == pytest.approx(1.0, rel=1e-12)


def test_drifting_above_the_reference_policy_is_penalised():
    drift = kamath_ch5_reward_kl_penalty("p", "resp", 0.5, 0.25, 2.0, r_theta=1.0)["estimate"]
    stay = kamath_ch5_reward_kl_penalty("p", "resp", 0.25, 0.25, 2.0, r_theta=1.0)["estimate"]
    assert drift < stay
