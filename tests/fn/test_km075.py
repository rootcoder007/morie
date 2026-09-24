"""Verification tests for km075.

Kamath, Keenan, Somers and Sorenson (2024), ch 5, the simplified DPO preference, eq. 5.11. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km075 import kamath_ch5_dpo_pref_simplified


def test_the_simplified_preference_is_the_sigmoid_of_the_reward_gap():
    res = kamath_ch5_dpo_pref_simplified([0.75, 0.25], [0.5, 0.5], 1.0)
    logit = math.log(0.75 / 0.5) - math.log(0.25 / 0.5)
    assert logit == pytest.approx(math.log(3.0), rel=1e-12)
    assert res["estimate"] == pytest.approx(1.0 / (1.0 + math.exp(-logit)),
                                            rel=1e-12)
    assert round(res["estimate"], 12) == pytest.approx(0.75, abs=1e-12)


def test_an_unchanged_policy_leaves_the_two_responses_indifferent():
    assert kamath_ch5_dpo_pref_simplified([0.5, 0.5], [0.5, 0.5], 2.0)["estimate"] == pytest.approx(
        0.5, rel=1e-12)


def test_a_larger_coefficient_sharpens_the_preference():
    assert kamath_ch5_dpo_pref_simplified([0.75, 0.25], [0.5, 0.5], 2.0)["estimate"] == pytest.approx(
        0.9, rel=1e-12)


def test_it_agrees_with_the_form_that_carries_the_partition_function():
    from morie.fn.km074 import kamath_ch5_dpo_pref_substituted as subst
    plain = kamath_ch5_dpo_pref_simplified([0.75, 0.25], [0.5, 0.5], 1.5)["estimate"]
    withz = subst([0.75, 0.25], [0.5, 0.5], 1.5, Z=17.0)["estimate"]
    assert plain == pytest.approx(withz, rel=1e-12)
