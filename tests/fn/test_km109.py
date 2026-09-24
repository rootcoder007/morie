"""Verification tests for km109.

Kamath, Keenan, Somers and Sorenson (2024), ch 6, the perplexity-ratio leakage score, eq. 6.33. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km109 import kamath_ch6_perplexity_leakage


def test_the_leakage_score_is_the_largest_log_perplexity_ratio():
    # log(PP_public(w) / PP_lm(w)) over the unique sequences
    res = kamath_ch6_perplexity_leakage(["w1", "w2"], {"w1": 10.0, "w2": 4.0},
               {"w1": 5.0, "w2": 4.0})
    assert res["estimate"] == pytest.approx(math.log(10.0 / 5.0), rel=1e-12)
    assert res["estimate"] == pytest.approx(math.log(2.0), rel=1e-12)
    assert res["argmax"] == "w1"


def test_a_model_no_more_surprised_than_the_public_baseline_leaks_nothing():
    res = kamath_ch6_perplexity_leakage(["w1"], {"w1": 4.0}, {"w1": 4.0})
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_memorising_a_sequence_raises_its_score_above_the_others():
    res = kamath_ch6_perplexity_leakage(["plain", "memorised"],
               {"plain": 4.0, "memorised": 100.0},
               {"plain": 4.0, "memorised": 2.0})
    assert res["argmax"] == "memorised"
    assert res["estimate"] == pytest.approx(math.log(50.0), rel=1e-12)
