"""Verification tests for the_r_series_dick_j_brus_spatial_sampling_with_r10e13.

Brus (2022), Spatial Sampling with R, eq. (10.13), the variance of the regression estimator. Expected values are
recomputed from the formula in the test body.
"""

import math

import pytest

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r10e13 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_13


def test_variance_of_the_regression_estimator():
    # (10.13): V = (1 - n/N) S2(e)/n with S2(e) = sum e^2/(n - 1)
    e = [0.5, -1.2, 0.3, 0.8, -0.4]
    n, N = 5, 100
    s2 = sum(v ** 2 for v in e) / (n - 1)
    res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_13(e, n, N)
    assert res["variance"] == pytest.approx((1.0 - n / N) * s2 / n, rel=1e-12)


def test_a_census_has_no_sampling_variance():
    e = [0.5, -1.2, 0.3, 0.8]
    res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_13(e, 4, 4)
    assert res["variance"] == pytest.approx(0.0, abs=1e-15)


def test_the_finite_population_correction_lowers_the_variance():
    e = [0.5, -1.2, 0.3, 0.8, -0.4]
    small_pop = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_13(e, 5, 10)["variance"]
    large_pop = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_13(e, 5, 100000)["variance"]
    assert small_pop < large_pop
