"""Verification tests for km140.

Kamath, Keenan, Somers and Sorenson (2024), eq. 9.12, the masked object-classification loss. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km140 import kamath_ch9_moc_loss


def test_the_masked_object_loss_sums_the_cross_entropies():
    # Eq 9.12: L_MOC = sum_i CE(c(v_m^i), g_theta(v_m^i))
    res = kamath_ch9_moc_loss(None, None, [[0.0]], [[0.5, 0.5], [0.25, 0.75]], labels=[0, 1])
    expected = math.log(2.0) - math.log(0.75)
    assert res["estimate"] == pytest.approx(expected, rel=1e-12)


def test_a_confident_correct_prediction_costs_almost_nothing():
    res = kamath_ch9_moc_loss(None, None, [[0.0]], [[1.0, 0.0]], labels=[0])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_predicting_the_wrong_class_is_expensive():
    good = kamath_ch9_moc_loss(None, None, [[0.0]], [[0.9, 0.1]], labels=[0])["estimate"]
    bad = kamath_ch9_moc_loss(None, None, [[0.0]], [[0.1, 0.9]], labels=[0])["estimate"]
    assert bad > good
