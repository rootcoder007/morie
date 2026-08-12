"""Tests for pcm (generalized partial credit model).

Replaces the generated stub, which imported a name the module never had.
"""

import math

from morie.fn.pcm import pcm


def test_the_probabilities_are_a_distribution():
    res = pcm(0.5, [-1.0, 0.0, 1.0])
    p = res["probabilities"]
    assert res["n_categories"] == 4
    assert len(p) == 4                      # k steps give k + 1 categories
    assert abs(sum(p) - 1.0) < 1e-12
    assert all(v > 0 for v in p)


def test_at_a_step_the_two_adjacent_categories_are_equally_likely():
    # theta equal to a step difficulty makes P(category j) = P(j - 1)
    res = pcm(0.0, [0.0])
    assert abs(res["probabilities"][0] - res["probabilities"][1]) < 1e-12


def test_higher_ability_shifts_mass_upward():
    low = pcm(-2.0, [-1.0, 0.0, 1.0])["probabilities"]
    high = pcm(2.0, [-1.0, 0.0, 1.0])["probabilities"]
    assert high[-1] > low[-1]
    assert low[0] > high[0]


def test_the_expected_score_increases_with_ability():
    steps = [-1.0, 0.0, 1.0]
    scores = [pcm(t, steps)["expected_score"] for t in (-3.0, -1.0, 1.0, 3.0)]
    assert all(scores[i] < scores[i + 1] for i in range(len(scores) - 1))
    assert 0.0 <= scores[0] and scores[-1] <= 3.0


def test_the_expected_score_matches_the_probabilities():
    res = pcm(0.3, [-0.5, 0.5])
    want = sum(j * res["probabilities"][j] for j in range(len(res["probabilities"])))
    assert abs(res["expected_score"] - want) < 1e-12


def test_discrimination_sharpens_the_curve():
    flat = pcm(1.0, [0.0], a=0.3)["probabilities"][1]
    steep = pcm(1.0, [0.0], a=3.0)["probabilities"][1]
    assert steep > flat


def test_one_step_reduces_to_the_two_parameter_logistic():
    a, b, theta = 1.2, 0.4, 0.9
    res = pcm(theta, [b], a=a, D=1.0)
    want = 1.0 / (1.0 + math.exp(-a * (theta - b)))
    assert abs(res["probabilities"][1] - want) < 1e-12
