"""Verification tests for the_r_series_dick_j_brus_spatial_sampling_with_r2e3.

Brus (2022), Spatial Sampling with R, eq. (2.3), the design weight. Expected values are
recomputed from the formula in the test body.
"""

import math

import pytest

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r2e3 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_2_equation_3


def test_design_weight_is_the_reciprocal_inclusion_probability():
    # (2.3): w_k = 1/pi_k
    for pi in (0.5, 0.25, 0.1, 1.0):
        assert the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_2_equation_3(pi)["value"] == pytest.approx(1.0 / pi, rel=1e-12)


def test_a_certainty_unit_carries_unit_weight():
    assert the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_2_equation_3(1.0)["value"] == pytest.approx(1.0, rel=1e-12)


def test_the_weights_of_an_equal_probability_sample_expand_to_the_population():
    # under simple random sampling pi = n/N, so each of the n weights is
    # N/n and they sum to N
    n, N = 20, 500
    w = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_2_equation_3(n / N)["value"]
    assert n * w == pytest.approx(float(N), rel=1e-12)
