"""Verification tests for km103.

Kamath, Keenan, Somers and Sorenson (2024), eq. 6.27, the LSTM word softmax. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km103 import kamath_ch6_lstm_softmax_word


def test_the_word_softmax_normalises_the_projected_hidden_state():
    # Eq 6.27: P(w_t = i) = softmax(U_i . f(c) + b_i)
    U = [[1.0, 0.0], [0.0, 1.0]]
    res = kamath_ch6_lstm_softmax_word(U, None, [1.0, 0.0], [0.0, 0.0])
    logits = [1.0, 0.0]
    total = sum(math.exp(v) for v in logits)
    assert res["p"][0] == pytest.approx(math.exp(1.0) / total, rel=1e-10)
    assert round(res["p"][0], 10) == pytest.approx(0.7310585786, abs=1e-10)
    assert res["argmax"] == 0
    assert sum(res["p"]) == pytest.approx(1.0, rel=1e-12)


def test_the_bias_shifts_which_word_wins():
    U = [[1.0, 0.0], [0.0, 1.0]]
    res = kamath_ch6_lstm_softmax_word(U, None, [1.0, 0.0], [0.0, 5.0])
    assert res["argmax"] == 1


def test_equal_logits_give_a_uniform_distribution():
    U = [[1.0, 0.0], [1.0, 0.0]]
    res = kamath_ch6_lstm_softmax_word(U, None, [1.0, 0.0], [0.0, 0.0])
    for p in res["p"]:
        assert p == pytest.approx(0.5, rel=1e-12)
