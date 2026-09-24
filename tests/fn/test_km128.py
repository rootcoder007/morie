"""Verification tests for km128.

Kamath, Keenan, Somers and Sorenson (2024), Large Language Models: A
Deep Dive, eq. 8.16, the unbiased pass@k. Expected values are recomputed in the test body.
"""

import math

import pytest

from morie.fn.km128 import kamath_ch8_pass_at_k


def test_pass_at_k_is_one_minus_the_all_fail_probability():
    # Eq 8.16: pass@k = 1 - C(n - c, k)/C(n, k)
    for n, c, k in ((10, 3, 2), (20, 1, 5), (8, 4, 4)):
        res = kamath_ch8_pass_at_k(n, c, k)
        expected = 1.0 - math.comb(n - c, k) / math.comb(n, k)
        assert res["estimate"] == pytest.approx(expected, rel=1e-12)
        assert res["fail_probability"] == pytest.approx(1.0 - expected, rel=1e-12)


def test_no_correct_sample_can_never_pass():
    assert kamath_ch8_pass_at_k(10, 0, 3)["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_all_correct_samples_always_pass():
    assert kamath_ch8_pass_at_k(10, 10, 3)["estimate"] == pytest.approx(1.0, rel=1e-12)


def test_drawing_more_samples_cannot_lower_the_pass_rate():
    one = kamath_ch8_pass_at_k(10, 3, 1)["estimate"]
    three = kamath_ch8_pass_at_k(10, 3, 3)["estimate"]
    assert three >= one
