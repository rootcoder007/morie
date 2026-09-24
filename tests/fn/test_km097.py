"""Verification tests for km097.

Kamath, Keenan, Somers and Sorenson (2024), the entropy attention regulariser. Expected values are
recomputed in the test body, and the value the docstring quotes is
asserted as well.
"""

import math

import pytest

from morie.fn.km097 import kamath_ch6_ear_entropy_reg


def test_the_entropy_regulariser_is_the_negated_summed_entropy():
    # R = -lam sum_l entropy(A)_l, so minimising R maximises entropy
    res = kamath_ch6_ear_entropy_reg([[[0.5, 0.5]]], lam=1.0)
    assert res["estimate"] == pytest.approx(-math.log(2.0), rel=1e-12)


def test_a_peaked_attention_row_contributes_no_entropy():
    res = kamath_ch6_ear_entropy_reg([[[1.0, 0.0]]])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_the_weight_scales_the_penalty():
    one = kamath_ch6_ear_entropy_reg([[[0.5, 0.5]]], lam=1.0)["estimate"]
    three = kamath_ch6_ear_entropy_reg([[[0.5, 0.5]]], lam=3.0)["estimate"]
    assert three == pytest.approx(3.0 * one, rel=1e-12)


def test_spreading_attention_lowers_the_objective():
    # more entropy means a more negative R
    peaked = kamath_ch6_ear_entropy_reg([[[0.9, 0.1]]])["estimate"]
    spread = kamath_ch6_ear_entropy_reg([[[0.5, 0.5]]])["estimate"]
    assert spread < peaked
