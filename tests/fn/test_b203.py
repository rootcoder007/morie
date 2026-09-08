"""Tests for b203.burkov_lm_ch2_lm_shorthand."""

from morie.fn.b203 import burkov_lm_ch2_lm_shorthand


def test_b203_basic():
    assert burkov_lm_ch2_lm_shorthand("b", ["a", "b", "a"])["notations_agree"] is True


def test_b203_edge():
    out = burkov_lm_ch2_lm_shorthand("b", ["a", "b", "a"])
    assert out["via_sequence"] == out["via_tokens"]
