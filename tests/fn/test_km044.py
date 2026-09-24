"""Verification tests for km044.

Kamath, Keenan, Somers and Sorenson (2024), eq. 3.3, the argmax answer search. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km044 import kamath_ch3_prompt_search_argmax


def test_the_search_returns_the_highest_scoring_answer():
    # Eq 3.3: z_hat = argmax_z P(f_fill(x, z); theta)
    scores = {"good": 0.9, "bad": 0.2, "fine": 0.5}
    res = kamath_ch3_prompt_search_argmax("x", list(scores), lambda filled: scores[filled.split("|")[-1]],
               f_fill=lambda x, z: x + "|" + z)
    assert res["z_hat"] == "good"
    assert res["estimate"] == pytest.approx(0.9, rel=1e-12)
    for z, v in scores.items():
        assert res["scores"][z] == pytest.approx(v, rel=1e-12)


def test_every_candidate_is_scored():
    res = kamath_ch3_prompt_search_argmax("x", ["a", "b", "c"], lambda s: len(s) / 10.0)
    assert res["n"] == 3
    assert set(res["scores"]) == {"a", "b", "c"}


def test_a_single_candidate_is_its_own_argmax():
    res = kamath_ch3_prompt_search_argmax("x", ["only"], lambda s: 0.42)
    assert res["z_hat"] == "only"
    assert res["estimate"] == pytest.approx(0.42, rel=1e-12)
