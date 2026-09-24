"""Verification tests for km114.

Kamath, Keenan, Somers and Sorenson (2024), Large Language Models: A
Deep Dive, eq. 8.2, the clipped n-gram precision of BLEU. Expected values are recomputed in the test body.
"""

import math

import pytest

from morie.fn.km114 import kamath_ch8_bleu_precision


def test_clipped_precision_per_order():
    # Eq 8.2: p_n = clipped matches / generated n-grams of that order
    grams = [[3, 4], [2, 3], [1, 2]]
    res = kamath_ch8_bleu_precision(grams)
    expected = [m / t for m, t in grams]
    for got, want in zip(res["p_n"], expected):
        assert got == pytest.approx(want, rel=1e-12)
    assert res["n"] == 3


def test_a_perfect_candidate_has_precision_one():
    res = kamath_ch8_bleu_precision([[5, 5], [4, 4]])
    assert all(v == pytest.approx(1.0, rel=1e-12) for v in res["p_n"])


def test_no_matches_gives_zero_precision():
    res = kamath_ch8_bleu_precision([[0, 6]])
    assert res["p_n"][0] == pytest.approx(0.0, abs=1e-15)
