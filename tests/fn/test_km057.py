"""Verification tests for km057.

Kamath, Keenan, Somers and Sorenson (2024), eq. 4.3, the LoRA objective. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km057 import kamath_ch4_lora_obj


def test_the_lora_objective_is_the_same_sum_as_full_finetuning():
    # Eq 4.3: the SAME token log-probability sum as Eq 4.2, taken over
    # the low-rank parameters instead of every weight
    theta = lambda x, prefix, y_t: 0.5
    base = lambda x, prefix, y_t: 0.25
    res = kamath_ch4_lora_obj(theta, base, ["x1"], [["a", "b"]])
    assert res["estimate"] == pytest.approx(2.0 * math.log(0.5), rel=1e-12)


def test_the_frozen_base_does_not_enter_the_objective():
    # only the adapted model scores the tokens; the base is carried for
    # reference, so changing it must not move the objective
    theta = lambda x, prefix, y_t: 0.5
    a = kamath_ch4_lora_obj(theta, lambda *_: 0.1, ["x1"], [["a", "b"]])["estimate"]
    b = kamath_ch4_lora_obj(theta, lambda *_: 0.9, ["x1"], [["a", "b"]])["estimate"]
    assert a == pytest.approx(b, rel=1e-12)


def test_a_certain_adapted_model_reaches_zero():
    res = kamath_ch4_lora_obj(lambda *_: 1.0, lambda *_: 0.5, ["x1"], [["a"]])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)
