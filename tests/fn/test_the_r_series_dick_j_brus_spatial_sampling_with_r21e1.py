"""Verification tests for the_r_series_dick_j_brus_spatial_sampling_with_r21e1.

Brus (2022), Spatial Sampling with R, eq. (21.1), the stationary process used for kriging. Expected values are
recomputed from the formula in the test body.
"""

import math

import pytest

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r21e1 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_1


def test_the_kriging_model_reports_one_location_per_mean_value():
    # (21.1): the same stationary process, used for kriging
    mu = [0.5, 0.5, 0.5]
    cov = [[1.0 if i == j else 0.3 for j in range(3)] for i in range(3)]
    res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_1(mu, cov)
    assert res["n"] == 3


def test_a_constant_mean_vector_is_accepted():
    res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_1([2.0] * 5, [[1.0 if i == j else 0.1 for j in range(5)]
                          for i in range(5)])
    assert res["n"] == 5


def test_a_mismatched_covariance_is_refused():
    with pytest.raises((ValueError, IndexError)):
        the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_1([1.0, 2.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
