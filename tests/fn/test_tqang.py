"""Tests for tqang (angle quantisation).

Replaces the generated stub, which imported
``turboquant_angle_quantization``.
"""

import math

from morie.fn.tqang import angular_difference, quantize_angles, wrap_angle


def test_wrapping_lands_in_the_half_open_interval():
    for t in (0.0, math.pi, -math.pi, 3 * math.pi, -3.5 * math.pi):
        w = wrap_angle(t)
        assert -math.pi <= w < math.pi


def test_wrapping_preserves_the_angle_modulo_two_pi():
    for t in (0.3, 7.0, -9.4):
        assert abs(math.cos(wrap_angle(t)) - math.cos(t)) < 1e-12
        assert abs(math.sin(wrap_angle(t)) - math.sin(t)) < 1e-12


def test_angular_difference_takes_the_short_way_round():
    d = angular_difference(0.1, 2 * math.pi - 0.1)
    assert abs(d - 0.2) < 1e-12
    assert abs(angular_difference(math.pi / 2, 0.0) - math.pi / 2) < 1e-12


def test_quantisation_error_never_exceeds_half_a_sector():
    thetas = [(-math.pi + i * 0.05) for i in range(120)]
    res = quantize_angles(thetas, bits=4)
    assert res["levels"] == 16
    assert abs(res["delta"] - 2 * math.pi / 16) < 1e-12
    assert res["max_abs_error"] <= res["half_delta"] + 1e-12


def test_more_bits_quantise_more_finely():
    thetas = [(-math.pi + i * 0.017) for i in range(300)]
    coarse = quantize_angles(thetas, bits=3)
    fine = quantize_angles(thetas, bits=7)
    assert fine["mse"] < coarse["mse"]
    assert fine["delta"] < coarse["delta"]


def test_the_mse_sits_at_the_uniform_bound():
    # delta^2/12 is the variance of a uniform error, so it is what the
    # empirical MSE approaches from either side on a finite sample --
    # not a hard ceiling. Measured here at 0.07% above it.
    thetas = [(-math.pi + i * 0.01) for i in range(600)]
    res = quantize_angles(thetas, bits=5)
    assert abs(res["mse"] / res["mse_bound"] - 1.0) < 0.05


def test_indices_stay_inside_the_codebook():
    res = quantize_angles([0.0, 3.0, -3.0, 1.5], bits=4)
    assert all(0 <= i < 16 for i in res["indices"])


def test_validation():
    for call in (lambda: quantize_angles([0.0], bits=0),
                 lambda: quantize_angles([0.0], bits=31)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
