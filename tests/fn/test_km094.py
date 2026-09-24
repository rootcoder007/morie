"""Verification tests for km094.

Kamath, Keenan, Somers and Sorenson (2024), the counterfactual-pair debiasing regulariser. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km094 import kamath_ch6_debias_regularizer


def test_the_debias_regulariser_is_the_weighted_squared_pair_distance():
    # Eq 6.x: R = lam sum ||E(a_i) - E(a_j)||^2
    pairs = [("he", "she"), ("man", "woman")]
    emb = {"he": [1.0, 0.0], "she": [0.0, 1.0],
           "man": [2.0, 0.0], "woman": [0.0, 2.0]}
    lam = 0.5
    res = kamath_ch6_debias_regularizer(pairs, emb, lam)
    total = sum(sum((emb[a][k] - emb[b][k]) ** 2 for k in range(2))
                for a, b in pairs)
    assert res["unweighted"] == pytest.approx(total, rel=1e-12)
    assert res["estimate"] == pytest.approx(lam * total, rel=1e-12)


def test_a_collapsed_pair_costs_nothing():
    emb = {"a": [1.0, 2.0], "b": [1.0, 2.0]}
    res = kamath_ch6_debias_regularizer([("a", "b")], emb, 1.0)
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_the_penalty_scales_linearly_with_its_weight():
    pairs = [("he", "she")]
    emb = {"he": [1.0, 0.0], "she": [0.0, 1.0]}
    one = kamath_ch6_debias_regularizer(pairs, emb, 1.0)["estimate"]
    three = kamath_ch6_debias_regularizer(pairs, emb, 3.0)["estimate"]
    assert three == pytest.approx(3.0 * one, rel=1e-12)
