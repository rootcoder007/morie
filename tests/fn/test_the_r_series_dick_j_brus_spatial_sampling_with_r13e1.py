"""Verification tests for the_r_series_dick_j_brus_spatial_sampling_with_r13e1.

Brus (2022), Spatial Sampling with R, eq. (13.1), the stationary Gaussian process model. Expected values are
recomputed from the formula in the test body.
"""

import math

import pytest

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r13e1 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_1


def test_the_model_reports_one_location_per_mean_value():
    # (13.1): Z(s) = mu(s) + eps(s) over the sampled locations
    mu = [1.0, 2.0, 3.0, 4.0]
    cov = [[1.0 if i == j else 0.2 for j in range(4)] for i in range(4)]
    res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_1(mu, cov)
    assert res["n"] == 4


def test_a_single_location_is_accepted():
    res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_1([2.0], [[1.0]])
    assert res["n"] == 1


def test_a_covariance_that_does_not_match_the_mean_is_refused():
    with pytest.raises((ValueError, IndexError)):
        the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_1([1.0, 2.0, 3.0], [[1.0, 0.0], [0.0, 1.0]])
