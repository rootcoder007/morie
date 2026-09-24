"""Verification tests for km116.

Kamath, Keenan, Somers and Sorenson (2024), Large Language Models: A
Deep Dive, eq. 8.4, the BLEU brevity penalty. Expected values are recomputed in the test body.
"""

import math

import pytest

from morie.fn.km116 import kamath_ch8_brevity_penalty


def test_brevity_penalty_both_branches():
    # Eq 8.4: BP = 1 when c > r, else exp(1 - r/c)
    long_enough = kamath_ch8_brevity_penalty(12.0, 10.0)
    assert long_enough["estimate"] == pytest.approx(1.0, rel=1e-12)
    assert long_enough["penalized"] is False
    short = kamath_ch8_brevity_penalty(8.0, 10.0)
    assert short["estimate"] == pytest.approx(math.exp(1.0 - 10.0 / 8.0), rel=1e-12)
    assert short["penalized"] is True


def test_equal_lengths_are_not_penalised_below_one():
    res = kamath_ch8_brevity_penalty(10.0, 10.0)
    assert res["estimate"] == pytest.approx(1.0, rel=1e-12)


def test_the_penalty_deepens_as_the_candidate_shortens():
    mild = kamath_ch8_brevity_penalty(9.0, 10.0)["estimate"]
    severe = kamath_ch8_brevity_penalty(4.0, 10.0)["estimate"]
    assert severe < mild < 1.0
