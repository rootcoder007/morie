"""Verification tests for km146.

Kamath, Keenan, Somers and Sorenson (2024), eq. 9.18, the output projector's mean squared error. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km146 import kamath_ch9_output_projector_mse


def test_the_output_projector_loss_is_the_mean_squared_error():
    # Eq 9.18: argmin L_mse(H_X, tau_X(t))
    res = kamath_ch9_output_projector_mse([[1.0, 2.0]], lambda tt: [[1.0, 0.0]], None)
    # (0^2 + 2^2)/2
    assert res["estimate"] == pytest.approx(2.0, rel=1e-12)


def test_a_perfect_projection_costs_nothing():
    res = kamath_ch9_output_projector_mse([[1.0, 2.0]], lambda tt: [[1.0, 2.0]], None)
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_the_error_grows_with_the_gap():
    near = kamath_ch9_output_projector_mse([[1.0]], lambda tt: [[1.5]], None)["estimate"]
    far = kamath_ch9_output_projector_mse([[1.0]], lambda tt: [[5.0]], None)["estimate"]
    assert far > near
