"""Verification tests for the_r_series_dick_j_brus_spatial_sampling_with_r10e21.

Brus (2022), Spatial Sampling with R, eq. (10.21), the combined regression estimator over strata. Expected values are
recomputed from the formula in the test body.
"""

import math

import pytest

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r10e21 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_21


def test_combined_regression_estimator_over_strata():
    # (10.21): zbar_cregr = zbar_pi + b (xbar - xbar_hat_pi)
    for zbar, b, xbar, xhat in ((4.0, 1.5, 9.0, 8.0), (10.0, -0.25, 2.0, 6.0)):
        res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_21(zbar, b, xbar, xhat)
        assert res["value"] == pytest.approx(zbar + b * (xbar - xhat), rel=1e-12)


def test_a_perfectly_calibrated_covariate_leaves_the_estimator_unchanged():
    assert the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_21(4.0, 1.5, 8.0, 8.0)["value"] == pytest.approx(4.0, rel=1e-12)
