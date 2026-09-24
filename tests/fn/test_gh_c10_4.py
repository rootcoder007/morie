"""Verification tests for gh_c10_4.

Ghosal and van der Vaart (2017), sec. 10.2.3, two-model adaptation.
"""

import math

import pytest

from morie.fn.gh_c10_4 import ghosal_two_model_adp


def test_the_posterior_concentrates_on_the_model_containing_the_truth():
    # sec. 10.2.3: Pi = pi_0 Pi_0 + (1 - pi_0) Pi_1, and the weight of
    # the model that contains the truth tends to one
    res = ghosal_two_model_adp(n=400, pi0=0.5, seed=1)
    assert 0.0 <= res["posterior_weight_small_model"] <= 1.0
    assert res["posterior_weight_small_model"] > 0.9
    assert res["small_model_wins"] is True


def test_the_weight_is_a_probability_for_any_prior_split():
    for pi0 in (0.1, 0.5, 0.9):
        w = ghosal_two_model_adp(n=200, pi0=pi0, seed=3)["posterior_weight_small_model"]
        assert 0.0 <= w <= 1.0
