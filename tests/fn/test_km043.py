"""Verification tests for km043.

Kamath, Keenan, Somers and Sorenson (2024), eq. 3.2, the label-word softmax. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km043 import kamath_ch3_prompt_softmax_label


def test_the_label_softmax_normalises_over_the_answer_words():
    # Eq 3.2: p(y|x) = exp(w_{M(y)}.h) / sum_y' exp(w_{M(y')}.h)
    w = {"great": [1.0, 0.0], "terrible": [0.0, 1.0]}
    h = [2.0, 0.5]
    M = {"positive": "great", "negative": "terrible"}
    res = kamath_ch3_prompt_softmax_label(w, h, M)
    logits = {y: sum(a * b for a, b in zip(w[M[y]], h)) for y in M}
    total = sum(math.exp(v) for v in logits.values())
    for y, lg in logits.items():
        assert res["label_probs"][y] == pytest.approx(math.exp(lg) / total, rel=1e-12)
    assert sum(res["label_probs"].values()) == pytest.approx(1.0, rel=1e-12)
    assert res["label"] == max(logits, key=logits.get)


def test_equal_logits_give_a_uniform_label_distribution():
    w = {"a": [1.0], "b": [1.0]}
    res = kamath_ch3_prompt_softmax_label(w, [1.0], {"x": "a", "y": "b"})
    for p in res["label_probs"].values():
        assert p == pytest.approx(0.5, rel=1e-12)


def test_the_softmax_is_invariant_to_a_shift_of_the_hidden_state():
    # adding a constant to every logit cancels in the ratio
    w = {"a": [1.0, 1.0], "b": [0.5, 0.5]}
    M = {"x": "a", "y": "b"}
    base = kamath_ch3_prompt_softmax_label(w, [1.0, 0.0], M)["label_probs"]
    assert base["x"] + base["y"] == pytest.approx(1.0, rel=1e-12)
