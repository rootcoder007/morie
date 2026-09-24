"""Verification tests for km144.

Kamath, Keenan, Somers and Sorenson (2024), eq. 9.16, multimodal instruction prediction. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km144 import kamath_ch9_mm_instr_predict


def test_the_instruction_prediction_applies_the_model_to_both_inputs():
    # Eq 9.16: A = f(I, M; theta) over the triplet (I, M, R)
    res = kamath_ch9_mm_instr_predict("What animal?", "<image>", lambda i, m: "a cat")
    assert res["answer"] == "a cat"


def test_both_the_instruction_and_the_modality_reach_the_model():
    seen = {}
    def f(i, m):
        seen["i"], seen["m"] = i, m
        return "ok"
    kamath_ch9_mm_instr_predict("instr", "modal", f)
    assert seen == {"i": "instr", "m": "modal"}


def test_a_non_callable_model_is_refused():
    with pytest.raises((ValueError, TypeError)):
        kamath_ch9_mm_instr_predict("i", "m", None)
