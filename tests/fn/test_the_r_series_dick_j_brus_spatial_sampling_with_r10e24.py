"""Verification tests for the_r_series_dick_j_brus_spatial_sampling_with_r10e24.

Brus (2022), Spatial Sampling with R, eq. (10.24), the heteroscedastic through-the-origin model. Expected values are
recomputed from the formula in the test body.
"""

import math

import pytest

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r10e24 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_24


def test_through_the_origin_model_prediction_and_variance():
    # (10.24): Z = beta x, with Var(eps) = sigma2 x
    for beta, x, s2 in ((2.0, 3.0, 0.5), (-1.5, 4.0, 2.0)):
        res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_24(beta, x, s2)
        assert res["prediction"] == pytest.approx(beta * x, rel=1e-12)
        assert res["variance"] == pytest.approx(s2 * x, rel=1e-12)


def test_the_line_passes_through_the_origin():
    res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_24(2.0, 0.0, 0.5)
    assert res["prediction"] == pytest.approx(0.0, abs=1e-15)
    assert res["variance"] == pytest.approx(0.0, abs=1e-15)


def test_the_variance_grows_in_proportion_to_the_covariate():
    a = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_24(2.0, 1.0, 0.5)["variance"]
    b = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_24(2.0, 3.0, 0.5)["variance"]
    assert b == pytest.approx(3.0 * a, rel=1e-12)
