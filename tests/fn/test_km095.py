"""Verification tests for km095.

Kamath, Keenan, Somers and Sorenson (2024), eq. 6.19, the gender direction of an embedding. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km095 import kamath_ch6_gender_direction


def test_the_gender_direction_is_the_mean_pair_displacement():
    # Eq 6.19: g = (1/|A|) sum (E(a_j) - E(a_i))
    pairs = [("he", "she"), ("man", "woman")]
    emb = {"he": [1.0, 0.0], "she": [0.0, 1.0],
           "man": [2.0, 0.0], "woman": [0.0, 2.0]}
    res = kamath_ch6_gender_direction(pairs, emb)
    expected = [
        sum(emb[b][k] - emb[a][k] for a, b in pairs) / len(pairs)
        for k in range(2)
    ]
    for got, want in zip(res["g"], expected):
        assert got == pytest.approx(want, rel=1e-12)
    assert res["norm"] == pytest.approx(
        math.sqrt(sum(v * v for v in expected)), rel=1e-12)


def test_identical_pairs_give_a_degenerate_zero_direction():
    emb = {"a": [1.0, 2.0], "b": [1.0, 2.0]}
    res = kamath_ch6_gender_direction([("a", "b")], emb)
    assert res["norm"] == pytest.approx(0.0, abs=1e-15)
    assert res["degenerate"] is True


def test_swapping_every_pair_reverses_the_direction():
    pairs = [("he", "she"), ("man", "woman")]
    emb = {"he": [1.0, 0.0], "she": [0.0, 1.0],
           "man": [2.0, 0.0], "woman": [0.0, 2.0]}
    forward = kamath_ch6_gender_direction(pairs, emb)["g"]
    backward = kamath_ch6_gender_direction([(b, a) for a, b in pairs], emb)["g"]
    for f, b in zip(forward, backward):
        assert f == pytest.approx(-b, rel=1e-12)
