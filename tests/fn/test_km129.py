"""Verification tests for km129.

Kamath, Keenan, Somers and Sorenson (2024), eq. 9.1, the modality encoder. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km129 import kamath_ch9_modality_encoder


def test_the_encoder_applies_the_modality_encoder_to_its_input():
    # Eq 9.1: F_X = ME_X(I_X), a composition rather than a formula
    res = kamath_ch9_modality_encoder([3.0, 4.0], lambda z: [z[0], z[1], 0.0])
    assert list(res["features"]) == [3.0, 4.0, 0.0]
    # the headline value is the feature norm: 3-4-5 triangle
    assert res["estimate"] == pytest.approx(5.0, rel=1e-12)


def test_the_identity_encoder_returns_its_input_unchanged():
    res = kamath_ch9_modality_encoder([1.0, 0.0], lambda z: list(z))
    assert list(res["features"]) == [1.0, 0.0]
    assert res["estimate"] == pytest.approx(1.0, rel=1e-12)


def test_a_non_callable_encoder_is_refused():
    with pytest.raises((ValueError, TypeError)):
        kamath_ch9_modality_encoder([1.0], None)
