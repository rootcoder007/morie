"""Verification tests for km113.

Kamath, Keenan, Somers and Sorenson (2024), eq. 8.1, perplexity. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km113 import kamath_ch8_perplexity


def test_perplexity_is_the_exponential_of_the_mean_negative_log_likelihood():
    # Eq 8.1: PPL = exp(-(1/N) sum log p)
    probs = [0.5, 0.25, 0.5, 0.125]
    res = kamath_ch8_perplexity(probs, p_theta=probs)
    nll = -sum(math.log(p) for p in probs) / len(probs)
    assert res["mean_nll"] == pytest.approx(nll, rel=1e-12)
    assert res["estimate"] == pytest.approx(math.exp(nll), rel=1e-12)


def test_a_uniform_model_over_k_tokens_has_perplexity_k():
    # every token predicted with probability 1/8 gives perplexity 8
    probs = [0.125] * 6
    res = kamath_ch8_perplexity(probs, p_theta=probs)
    assert res["estimate"] == pytest.approx(8.0, rel=1e-12)


def test_a_certain_model_has_perplexity_one():
    probs = [1.0, 1.0, 1.0]
    res = kamath_ch8_perplexity(probs, p_theta=probs)
    assert res["estimate"] == pytest.approx(1.0, rel=1e-12)


def test_a_worse_model_has_a_higher_perplexity():
    good = kamath_ch8_perplexity([0.8] * 5, p_theta=[0.8] * 5)["estimate"]
    bad = kamath_ch8_perplexity([0.2] * 5, p_theta=[0.2] * 5)["estimate"]
    assert bad > good
