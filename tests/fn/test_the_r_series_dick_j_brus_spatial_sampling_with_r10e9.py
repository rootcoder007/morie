"""Verification tests for the_r_series_dick_j_brus_spatial_sampling_with_r10e9.

Brus (2022), Spatial Sampling with R, eq. (10.9), the regression estimator in slope form. Expected values are
recomputed from the formula in the test body.
"""

import math

import pytest

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r10e9 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_9


def test_slope_form_sums_one_correction_per_covariate():
    # (10.9): zbar_regr = zbar_pi + sum_j b_j (xbar_j - xbar_hat_j)
    zbar = 6.0
    b = [0.5, -1.25, 2.0]
    xbar = [10.0, 4.0, 1.0]
    xhat = [9.0, 5.0, 1.5]
    res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_9(zbar, b, xbar, xhat)
    expected = zbar + sum(bj * (t - h) for bj, t, h in zip(b, xbar, xhat))
    assert res["value"] == pytest.approx(expected, rel=1e-12)


def test_a_single_covariate_reduces_to_the_simple_form():
    res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_9(6.0, [0.5], [10.0], [9.0])
    assert res["value"] == pytest.approx(6.0 + 0.5 * 1.0, rel=1e-12)


def test_covariates_already_on_target_contribute_nothing():
    res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_9(6.0, [0.5, 2.0], [10.0, 3.0], [10.0, 3.0])
    assert res["value"] == pytest.approx(6.0, rel=1e-12)
