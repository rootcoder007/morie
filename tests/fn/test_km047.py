"""Verification tests for km047.

Kamath, Keenan, Somers and Sorenson (2024), eq. 3.6, the translation prefix prompt. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km047 import kamath_ch3_translate_prefix_prompt


def test_the_translation_prompt_states_the_instruction_before_the_input():
    # Eq 3.6: the instruction and input precede the answer slot
    res = kamath_ch3_translate_prefix_prompt("the cat sat")
    assert res["prompt"].lower().startswith("translate")
    assert "the cat sat" in res["prompt"]
    assert res["slot_filled"] is False


def test_the_answer_slot_is_filled_at_the_end():
    res = kamath_ch3_translate_prefix_prompt("the cat sat", "le chat")
    assert "[z]" not in res["prompt"]
    assert res["prompt"].rstrip().endswith("le chat")


def test_the_token_count_matches_the_rendered_prompt():
    res = kamath_ch3_translate_prefix_prompt("the cat sat", "le chat")
    assert res["n"] == len(res["prompt"].split())
