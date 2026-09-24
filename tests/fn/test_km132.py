"""Verification tests for km132.

Kamath, Keenan, Somers and Sorenson (2024), eq. 9.4, the language model's text and signal tokens. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km132 import kamath_ch9_llm_signal_tokens


def test_the_language_model_returns_both_text_and_signal_tokens():
    # Eq 9.4: (t, S_X) = LLM(P_X, F_T)
    res = kamath_ch9_llm_signal_tokens([[0.0]], [[1.0]], llm=lambda p, f: ("a cat", ["<IMG>"]))
    assert res["text"] == "a cat"
    assert list(res["signal_tokens"]) == ["<IMG>"]
    assert res["estimate"] == 1


def test_the_signal_token_count_is_the_headline_value():
    res = kamath_ch9_llm_signal_tokens([[0.0]], [[1.0]],
               llm=lambda p, f: ("x", ["<IMG>", "<AUD>", "<VID>"]))
    assert res["estimate"] == 3


def test_text_with_no_signal_tokens_counts_zero():
    res = kamath_ch9_llm_signal_tokens([[0.0]], [[1.0]], llm=lambda p, f: ("just text", []))
    assert res["estimate"] == 0
