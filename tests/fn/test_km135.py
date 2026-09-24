"""Verification tests for km135.

Kamath, Keenan, Somers and Sorenson (2024), eq. 9.7, the total CLIP contrastive loss. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km135 import kamath_ch9_clip_contrastive_total


def test_the_clip_loss_is_the_sum_of_both_directions():
    # Eq 9.7: L_CL = L_i2t + L_t2i
    for a, b in ((1.5, 2.5), (0.0, 0.0), (3.25, 0.75)):
        res = kamath_ch9_clip_contrastive_total(a, b)
        assert res["estimate"] == pytest.approx(a + b, rel=1e-12)


def test_the_loss_is_symmetric_in_its_two_halves():
    assert kamath_ch9_clip_contrastive_total(1.5, 2.5)["estimate"] == pytest.approx(
        kamath_ch9_clip_contrastive_total(2.5, 1.5)["estimate"], rel=1e-12)


def test_a_negative_cross_entropy_is_refused():
    # both halves are cross-entropies and so cannot be negative
    with pytest.raises(ValueError):
        kamath_ch9_clip_contrastive_total(-1.0, 2.0)
