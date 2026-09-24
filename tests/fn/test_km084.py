"""Verification tests for km084.

Kamath, Keenan, Somers and Sorenson (2024), Large Language Models: A
Deep Dive, eq. 6.8, the Log-Probability Bias Score. Expected values are recomputed in the test body.
"""

import math

import pytest

from morie.fn.km084 import kamath_ch6_lpbs_bias


def test_log_probability_bias_score_is_the_difference_of_normalised_logs():
    # Eq 6.8: LPBS = log(p_ai/p_prior_i) - log(p_aj/p_prior_j)
    p_a = [0.3, 0.7]
    p_prior = [0.5, 0.5]
    res = kamath_ch6_lpbs_bias(p_a, p_prior)
    li = math.log(p_a[0] / p_prior[0])
    lj = math.log(p_a[1] / p_prior[1])
    assert res["normalised_log_i"] == pytest.approx(li, rel=1e-12)
    assert res["normalised_log_j"] == pytest.approx(lj, rel=1e-12)
    assert res["estimate"] == pytest.approx(li - lj, rel=1e-12)


def test_an_unbiased_pair_scores_zero():
    res = kamath_ch6_lpbs_bias([0.5, 0.5], [0.5, 0.5])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_the_score_changes_sign_when_the_groups_swap():
    a = kamath_ch6_lpbs_bias([0.3, 0.7], [0.5, 0.5])["estimate"]
    b = kamath_ch6_lpbs_bias([0.7, 0.3], [0.5, 0.5])["estimate"]
    assert a == pytest.approx(-b, rel=1e-12)
