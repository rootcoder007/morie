"""Verification tests for km106.

Kamath, Keenan, Somers and Sorenson (2024), eq. 6.30, the self-diagnosis probability. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km106 import kamath_ch6_self_diagnosis_prob


def test_self_diagnosis_renormalises_over_yes_and_no():
    # Eq 6.30: p(y|x) = p("Yes") / (p("Yes") + p("No"))
    res = kamath_ch6_self_diagnosis_prob("I will hunt you down!", "a threat",
               lambda prompt: {"Yes": 0.3, "No": 0.1})
    assert res["estimate"] == pytest.approx(0.75, rel=1e-12)


def test_the_diagnosis_prompt_names_the_attribute():
    res = kamath_ch6_self_diagnosis_prob("some text", "a threat",
               lambda prompt: {"Yes": 0.5, "No": 0.5})
    assert "Does the above text contain a threat?" in res["prompt"]


def test_an_even_answer_gives_one_half():
    res = kamath_ch6_self_diagnosis_prob("x", "y", lambda prompt: {"Yes": 0.2, "No": 0.2})
    assert res["estimate"] == pytest.approx(0.5, rel=1e-12)
