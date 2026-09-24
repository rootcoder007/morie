"""Verification tests for the_r_series_dick_j_brus_spatial_sampling_with_r26e4.

Brus (2022), Spatial Sampling with R, eq. (26.4), the effective sample size under autocorrelation. Expected values are
recomputed from the formula in the test body.
"""

import math

import pytest

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r26e4 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_equation_4


def test_effective_sample_size_under_autocorrelation():
    # (26.4): n_eff = n/(1 + (n - 1) rhobar)
    for n, rho in ((10, 0.3), (50, 0.05), (4, 0.9)):
        res = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_equation_4(n, rho)
        assert res["value"] == pytest.approx(n / (1.0 + (n - 1) * rho), rel=1e-12)


def test_independent_observations_count_in_full():
    assert the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_equation_4(25, 0.0)["value"] == pytest.approx(25.0, rel=1e-12)


def test_perfect_correlation_leaves_one_effective_observation():
    assert the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_equation_4(25, 1.0)["value"] == pytest.approx(1.0, rel=1e-12)


def test_stronger_correlation_costs_effective_observations():
    weak = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_equation_4(30, 0.1)["value"]
    strong = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_equation_4(30, 0.5)["value"]
    assert strong < weak < 30.0
