"""Verification tests for km046.

Kamath, Keenan, Somers and Sorenson (2024), eq. 3.5, the prefix prompt template. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km046 import kamath_ch3_prefix_prompt_template


def test_the_prefix_template_puts_the_answer_slot_last():
    # Eq 3.5: x' = [x] This movie is [z]
    res = kamath_ch3_prefix_prompt_template("a fine film")
    assert res["prompt"].startswith("a fine film")
    assert res["prompt"].rstrip().endswith("[z]")
    assert res["slot_filled"] is False


def test_filling_the_slot_replaces_the_placeholder():
    res = kamath_ch3_prefix_prompt_template("a fine film", "great")
    assert "[z]" not in res["prompt"]
    assert res["prompt"].rstrip().endswith("great")
    assert res["slot_filled"] is True


def test_the_token_count_matches_the_rendered_prompt():
    res = kamath_ch3_prefix_prompt_template("a fine film", "great")
    assert res["n"] == len(res["prompt"].split())
    assert res["estimate"] == pytest.approx(float(res["n"]), rel=1e-12)
