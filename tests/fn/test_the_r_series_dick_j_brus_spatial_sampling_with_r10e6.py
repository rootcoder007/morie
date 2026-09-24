"""Verification tests for the_r_series_dick_j_brus_spatial_sampling_with_r10e6.

Brus (2022), Spatial Sampling with R, eq. (10.6), the sample-weighted generalised least-squares slope. Expected values are
recomputed from the formula in the test body.
"""

import math

import pytest

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r10e6 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_6


def test_gls_slope_matches_the_weighted_normal_equations():
    # (10.6): b = (sum x x'/(sigma2 pi))^-1 sum x z/(sigma2 pi)
    x = [1.0, 2.0, 3.0, 4.0]
    z = [2.1, 3.9, 6.2, 7.8]
    s2 = [1.0, 1.0, 2.0, 2.0]
    pi = [0.5, 0.5, 0.25, 0.25]
    num = sum(xi * zi / (si * pj) for xi, zi, si, pj in zip(x, z, s2, pi))
    den = sum(xi * xi / (si * pj) for xi, si, pj in zip(x, s2, pi))
    res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_6(x, z, s2, pi)
    assert res["value"] == pytest.approx(num / den, rel=1e-10)


def test_an_exact_proportional_relation_is_recovered_exactly():
    x = [1.0, 2.0, 5.0]
    slope = 2.75
    z = [slope * v for v in x]
    res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_6(x, z, [1.0] * 3, [1.0] * 3)
    assert res["value"] == pytest.approx(slope, rel=1e-10)


def test_equal_weights_reduce_to_ordinary_through_the_origin_least_squares():
    x = [1.0, 2.0, 3.0]
    z = [1.0, 2.5, 2.8]
    ols = sum(a * b for a, b in zip(x, z)) / sum(a * a for a in x)
    res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_6(x, z, [1.0] * 3, [1.0] * 3)
    assert res["value"] == pytest.approx(ols, rel=1e-10)
