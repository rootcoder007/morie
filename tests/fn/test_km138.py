"""Verification tests for km138.

Kamath, Keenan, Somers and Sorenson (2024), eq. 9.10, the SimVLM masked language-modelling loss. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km138 import kamath_ch9_simvlm_mlm


def test_the_masked_loss_averages_over_the_masked_positions():
    # Eq 9.10: L_MLM = -E log P(x_m | x_not m, v)
    res = kamath_ch9_simvlm_mlm(None, [0.5, 1.0, 0.25], [[0.0]], [0, 2])
    expected = (math.log(2.0) + math.log(4.0)) / 2.0
    assert res["estimate"] == pytest.approx(expected, rel=1e-12)


def test_only_the_masked_positions_are_scored():
    # position 1 has probability one and is not masked, so masking it
    # instead must change the loss
    a = kamath_ch9_simvlm_mlm(None, [0.5, 1.0, 0.25], [[0.0]], [0, 2])["estimate"]
    b = kamath_ch9_simvlm_mlm(None, [0.5, 1.0, 0.25], [[0.0]], [1])["estimate"]
    assert b == pytest.approx(0.0, abs=1e-15)
    assert a > b


def test_a_certain_model_costs_nothing():
    res = kamath_ch9_simvlm_mlm(None, [1.0, 1.0], [[0.0]], [0, 1])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)
