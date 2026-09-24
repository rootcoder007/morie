"""Verification tests for km048.

Kamath, Keenan, Somers and Sorenson (2024), eq. 3.7, the cloze prompt template. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km048 import kamath_ch3_cloze_prompt_template


def test_the_cloze_template_has_tokens_on_both_sides_of_the_slot():
    # Eq 3.7: [x] This is a [z] movie. -- the slot is interior
    res = kamath_ch3_cloze_prompt_template("a fine film")
    prompt = res["prompt"]
    before, after = prompt.split("[z]")
    assert before.strip() != ""
    assert after.strip() != ""
    assert res["slot_filled"] is False


def test_filling_the_interior_slot_keeps_the_trailing_words():
    res = kamath_ch3_cloze_prompt_template("a fine film", "great")
    assert "great" in res["prompt"]
    assert res["prompt"].rstrip().endswith("movie.")
    assert res["slot_filled"] is True


def test_the_token_count_matches_the_rendered_prompt():
    res = kamath_ch3_cloze_prompt_template("a fine film", "great")
    assert res["n"] == len(res["prompt"].split())
