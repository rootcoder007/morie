"""Verification tests for km104.

Kamath, Keenan, Somers and Sorenson (2024), eq. 6.28, the affect language model. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km104 import kamath_ch6_affect_lm


def test_the_affect_model_adds_a_weighted_emotion_term():
    # Eq 6.28: softmax(U_i . f(c) + beta V_i . g(e) + b_i)
    U = [[1.0, 0.0], [0.0, 1.0]]
    V = [[0.0, 0.0], [1.0, 0.0]]
    res = kamath_ch6_affect_lm(U, V, None, None, [1.0, 0.0], [1.0, 0.0], 2.0, [0.0, 0.0])
    logits = [1.0 + 2.0 * 0.0, 0.0 + 2.0 * 1.0]
    total = sum(math.exp(v) for v in logits)
    assert res["p"][1] == pytest.approx(math.exp(2.0) / total, rel=1e-10)
    assert round(res["p"][1], 10) == pytest.approx(0.7310585786, abs=1e-10)


def test_a_zero_weight_reduces_to_the_plain_language_model():
    U = [[1.0, 0.0], [0.0, 1.0]]
    V = [[0.0, 0.0], [9.0, 0.0]]
    res = kamath_ch6_affect_lm(U, V, None, None, [1.0, 0.0], [1.0, 0.0], 0.0, [0.0, 0.0])
    assert res["argmax"] == 0


def test_raising_the_weight_strengthens_the_affect_term():
    U = [[1.0, 0.0], [0.0, 1.0]]
    V = [[0.0, 0.0], [1.0, 0.0]]
    mild = kamath_ch6_affect_lm(U, V, None, None, [1.0, 0.0], [1.0, 0.0], 0.5, [0.0, 0.0])["p"][1]
    strong = kamath_ch6_affect_lm(U, V, None, None, [1.0, 0.0], [1.0, 0.0], 4.0, [0.0, 0.0])["p"][1]
    assert strong > mild
