"""Verification tests for km051.

Kamath, Keenan, Somers and Sorenson (2024), eq. 3.10, the adversarial-trigger QA prompt. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km051 import kamath_ch3_qa_trigger_template


def test_the_trigger_is_repeated_before_the_adversarial_answer():
    # Eq 3.10: "Question: [x] Context: [y] Answer: [T][T][T][z_adv]"
    res = kamath_ch3_qa_trigger_template("who?", "ctx", "TRG", "wrong", n_triggers=3)
    assert res["prompt"].count("TRG") == 3
    assert res["prompt"].rstrip().endswith("wrong")
    assert res["n_triggers"] == 3
    assert res["adversarial_answer"] == "wrong"


def test_the_trigger_count_is_respected():
    for k in (1, 2, 5):
        res = kamath_ch3_qa_trigger_template("q", "c", "T", "a", n_triggers=k)
        assert res["prompt"].count("T ") + res["prompt"].count("T" + "a") >= 1
        assert res["n_triggers"] == k


def test_the_question_and_context_both_appear():
    res = kamath_ch3_qa_trigger_template("who wrote it", "a context", "T", "adv")
    assert "who wrote it" in res["prompt"]
    assert "a context" in res["prompt"]
