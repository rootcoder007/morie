"""Verification tests for km105.

Kamath, Keenan, Somers and Sorenson (2024), Large Language Models: A
Deep Dive, eq. 6.29, the GeDi combined loss. Expected values are recomputed in the test body.
"""

import math

import pytest

from morie.fn.km105 import kamath_ch6_gedi_combined_loss


def test_gedi_is_a_convex_combination_of_the_two_losses():
    # Eq 6.29: L_gd = lam L_g + (1 - lam) L_d
    for lg, ld, lam in ((2.0, 4.0, 0.25), (1.0, 1.0, 0.5), (3.0, 0.0, 0.9)):
        res = kamath_ch6_gedi_combined_loss(lg, ld, lam)
        assert res["estimate"] == pytest.approx(lam * lg + (1.0 - lam) * ld, rel=1e-12)


def test_the_endpoints_select_one_loss_each():
    assert kamath_ch6_gedi_combined_loss(2.0, 4.0, 1.0)["estimate"] == pytest.approx(2.0, rel=1e-12)
    assert kamath_ch6_gedi_combined_loss(2.0, 4.0, 0.0)["estimate"] == pytest.approx(4.0, rel=1e-12)


def test_the_result_lies_between_the_two_losses():
    res = kamath_ch6_gedi_combined_loss(2.0, 4.0, 0.3)
    assert 2.0 <= res["estimate"] <= 4.0


def test_a_weight_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError):
        kamath_ch6_gedi_combined_loss(2.0, 4.0, 1.5)
