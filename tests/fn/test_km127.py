"""Verification tests for km127.

Kamath, Keenan, Somers and Sorenson (2024), Large Language Models: A
Deep Dive, eq. 8.15, the G-Eval probability-weighted score. Expected values are recomputed in the test body.
"""

import math

import pytest

from morie.fn.km127 import kamath_ch8_geval_score


def test_geval_is_the_probability_weighted_score():
    # Eq 8.15: score = sum_i p(s_i) s_i
    s = [1, 2, 3]
    p = [0.2, 0.5, 0.3]
    res = kamath_ch8_geval_score(s, p)
    assert res["estimate"] == pytest.approx(sum(a * b for a, b in zip(s, p)), rel=1e-12)


def test_all_mass_on_one_score_returns_that_score():
    res = kamath_ch8_geval_score([1, 2, 3], [0.0, 1.0, 0.0])
    assert res["estimate"] == pytest.approx(2.0, rel=1e-12)


def test_the_score_lies_within_the_allowed_range():
    res = kamath_ch8_geval_score([1, 2, 3, 4, 5], [0.1, 0.2, 0.4, 0.2, 0.1])
    assert 1.0 <= res["estimate"] <= 5.0
