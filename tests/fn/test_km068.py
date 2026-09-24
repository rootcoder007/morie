"""Verification tests for km068.

Kamath, Keenan, Somers and Sorenson (2024), ch 5, the PPO objective, eq. 5.4. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km068 import kamath_ch5_ppo_loss


def test_the_ppo_objective_is_the_negated_expected_penalised_reward():
    r = lambda xi, yi: 1.0 if yi == "a" else 0.0
    res = kamath_ch5_ppo_loss([[0.5, 0.5]], ["p"], [["a", "b"]], r, 1.0,
               pi_ref=[[0.5, 0.5]])
    # E[r] = 0.5 * 1 + 0.5 * 0 with no divergence from the reference
    assert res["estimate"] == pytest.approx(-0.5, rel=1e-12)


def test_a_uniformly_zero_reward_gives_a_zero_objective():
    r = lambda xi, yi: 0.0
    res = kamath_ch5_ppo_loss([[0.5, 0.5]], ["p"], [["a", "b"]], r, 1.0,
               pi_ref=[[0.5, 0.5]])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_diverging_from_the_reference_policy_worsens_the_objective():
    r = lambda xi, yi: 0.0
    same = kamath_ch5_ppo_loss([[0.5, 0.5]], ["p"], [["a", "b"]], r, 1.0,
                pi_ref=[[0.5, 0.5]])["estimate"]
    apart = kamath_ch5_ppo_loss([[0.9, 0.1]], ["p"], [["a", "b"]], r, 1.0,
                 pi_ref=[[0.5, 0.5]])["estimate"]
    assert apart > same
