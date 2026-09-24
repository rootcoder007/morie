"""Verification tests for the_r_series_dick_j_brus_spatial_sampling_with_r10e10.

Brus (2022), Spatial Sampling with R, eq. (10.10), the simple regression estimator under simple random sampling. Expected values are
recomputed from the formula in the test body.
"""

import math

import pytest

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r10e10 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_10


def test_regression_estimator_adds_the_calibration_term():
    # (10.10): zbar_regr = zbar_S + b (xbar - xbar_S)
    for zbar, b, xbar, xbar_s in ((5.0, 0.8, 12.0, 10.0), (2.5, -0.4, 3.0, 7.0)):
        res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_10(zbar, b, xbar, xbar_s)
        assert res["value"] == pytest.approx(zbar + b * (xbar - xbar_s), rel=1e-12)


def test_a_zero_slope_leaves_the_sample_mean_alone():
    assert the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_10(5.0, 0.0, 12.0, 10.0)["value"] == pytest.approx(5.0, rel=1e-12)


def test_a_covariate_mean_already_on_target_needs_no_correction():
    assert the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_10(5.0, 0.8, 10.0, 10.0)["value"] == pytest.approx(5.0, rel=1e-12)
