"""Verification tests for km121.

Kamath, Keenan, Somers and Sorenson (2024), Large Language Models: A
Deep Dive, eq. 8.9, the BERTScore F1. Expected values are recomputed in the test body.
"""

import math

import pytest

from morie.fn.km121 import kamath_ch8_bertscore_f1


def test_bertscore_f1_is_the_harmonic_mean():
    # Eq 8.9: F = 2 P R / (P + R)
    for p, r in ((0.8, 0.6), (0.5, 0.5), (0.9, 0.1)):
        res = kamath_ch8_bertscore_f1(p, r)
        assert res["estimate"] == pytest.approx(2.0 * p * r / (p + r), rel=1e-12)


def test_equal_precision_and_recall_give_that_value():
    assert kamath_ch8_bertscore_f1(0.7, 0.7)["estimate"] == pytest.approx(0.7, rel=1e-12)


def test_the_harmonic_mean_never_exceeds_the_arithmetic_mean():
    p, r = 0.9, 0.2
    res = kamath_ch8_bertscore_f1(p, r)
    assert res["estimate"] <= (p + r) / 2.0 + 1e-15
