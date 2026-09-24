"""Verification tests for km056.

Kamath, Keenan, Somers and Sorenson (2024), eq. 4.2, the full-parameter fine-tuning objective. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km056 import kamath_ch4_full_finetune_obj


def test_the_full_finetune_objective_sums_log_probability_over_tokens():
    # Eq 4.2: sum over pairs of sum over t of log P_Phi(y_t | x, y_<t)
    model = lambda x, prefix, y_t: 0.5
    res = kamath_ch4_full_finetune_obj(model, ["x1"], [["a", "b"]])
    assert res["estimate"] == pytest.approx(2.0 * math.log(0.5), rel=1e-12)
    assert res["n_tokens"] == 2


def test_every_pair_contributes_its_own_sum():
    model = lambda x, prefix, y_t: 0.5
    one = kamath_ch4_full_finetune_obj(model, ["x1"], [["a", "b"]])["estimate"]
    two = kamath_ch4_full_finetune_obj(model, ["x1", "x2"], [["a", "b"], ["c", "d"]])["estimate"]
    assert two == pytest.approx(2.0 * one, rel=1e-12)


def test_a_certain_model_reaches_the_maximum_of_zero():
    model = lambda x, prefix, y_t: 1.0
    res = kamath_ch4_full_finetune_obj(model, ["x1"], [["a", "b"]])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)
