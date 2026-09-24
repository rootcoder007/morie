"""Verification tests for km117.

Kamath, Keenan, Somers and Sorenson (2024), Large Language Models: A
Deep Dive, eq. 8.5, the BLEU score. Expected values are recomputed in the test body.
"""

import math

import pytest

from morie.fn.km117 import kamath_ch8_bleu_final


def test_bleu_is_the_penalty_times_the_geometric_mean():
    # Eq 8.5: BLEU = BP exp(sum_n (1/N) log p_n)
    bp = 0.8
    p = [0.75, 0.5, 0.25, 0.125]
    res = kamath_ch8_bleu_final(bp, p)
    gm = math.exp(sum(math.log(v) for v in p) / len(p))
    assert res["geometric_mean"] == pytest.approx(gm, rel=1e-12)
    assert res["estimate"] == pytest.approx(bp * gm, rel=1e-12)


def test_uniform_precisions_make_the_geometric_mean_that_value():
    res = kamath_ch8_bleu_final(1.0, [0.5, 0.5, 0.5, 0.5])
    assert res["geometric_mean"] == pytest.approx(0.5, rel=1e-12)
    assert res["estimate"] == pytest.approx(0.5, rel=1e-12)


def test_a_single_zero_precision_zeroes_the_score():
    res = kamath_ch8_bleu_final(1.0, [0.5, 0.0, 0.5, 0.5])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)
