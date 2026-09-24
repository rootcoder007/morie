"""Verification tests for km143.

Kamath, Keenan, Somers and Sorenson (2024), eq. 9.15, the frame-order modelling loss. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km143 import kamath_ch9_fom_loss


def test_the_order_modelling_loss_sums_the_timestamp_log_probabilities():
    # Eq 9.15: L_FOM = -sum_i log P[r_i, t_i]
    res = kamath_ch9_fom_loss([0, 1], [0, 1], P=[[0.5, 0.5], [0.25, 0.75]])
    expected = math.log(2.0) - math.log(0.75)
    assert res["estimate"] == pytest.approx(expected, rel=1e-12)


def test_a_certain_ordering_costs_nothing():
    res = kamath_ch9_fom_loss([0], [0], P=[[1.0, 0.0]])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_each_frame_contributes_its_own_term():
    one = kamath_ch9_fom_loss([0], [0], P=[[0.5, 0.5]])["estimate"]
    two = kamath_ch9_fom_loss([0, 1], [0, 0], P=[[0.5, 0.5], [0.5, 0.5]])["estimate"]
    assert two == pytest.approx(2.0 * one, rel=1e-12)
