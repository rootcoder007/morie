"""Verification tests for km148.

Kamath, Keenan, Somers and Sorenson (2024), eq. 9.20, the latent diffusion loss. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km148 import kamath_ch9_ldm_loss


def test_the_diffusion_loss_is_the_squared_noise_residual():
    # Eq 9.20: L = E || eps - eps_X(z_t, t, H_X) ||^2
    res = kamath_ch9_ldm_loss([[1.0, 0.0]], [[0.0, 0.0]], [[0.0]],
               eps_net=lambda z, tt, h: [[0.0, 0.0]])
    assert res["estimate"] == pytest.approx(1.0, rel=1e-12)


def test_predicting_the_noise_exactly_costs_nothing():
    res = kamath_ch9_ldm_loss([[1.0, 0.0]], [[0.0, 0.0]], [[0.0]],
               eps_net=lambda z, tt, h: [[1.0, 0.0]])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_the_loss_is_quadratic_in_the_residual():
    one = kamath_ch9_ldm_loss([[1.0]], [[0.0]], [[0.0]],
               eps_net=lambda z, tt, h: [[0.0]])["estimate"]
    two = kamath_ch9_ldm_loss([[2.0]], [[0.0]], [[0.0]],
               eps_net=lambda z, tt, h: [[0.0]])["estimate"]
    assert two == pytest.approx(4.0 * one, rel=1e-12)
