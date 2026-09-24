"""Verification tests for the_r_series_dick_j_brus_spatial_sampling_with_r10e14.

Brus (2022), Spatial Sampling with R, eq. (10.14), the residual variance. Expected values are
recomputed from the formula in the test body.
"""

import math

import pytest

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r10e14 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_14


def test_residual_variance_divides_by_n_minus_one():
    # (10.14): S2(e) = sum e^2/(n - 1)
    e = [0.5, -1.2, 0.3, 0.8, -0.4]
    n = len(e)
    res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_14(e, n, 200)
    assert res["s2_e"] == pytest.approx(sum(v ** 2 for v in e) / (n - 1), rel=1e-12)


def test_zero_residuals_give_zero_variance():
    res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_14([0.0, 0.0, 0.0], 3, 50)
    assert res["s2_e"] == pytest.approx(0.0, abs=1e-15)


def test_scaling_the_residuals_scales_the_variance_quadratically():
    e = [0.5, -1.2, 0.3]
    one = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_14(e, 3, 50)["s2_e"]
    two = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_14([2.0 * v for v in e], 3, 50)["s2_e"]
    assert two == pytest.approx(4.0 * one, rel=1e-12)
