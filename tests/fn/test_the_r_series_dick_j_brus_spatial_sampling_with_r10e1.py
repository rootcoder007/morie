"""Verification tests for the_r_series_dick_j_brus_spatial_sampling_with_r10e1.

Brus (2022), Spatial Sampling with R, eq. (10.1), the model-assisted working model. Expected values are
recomputed from the formula in the test body.
"""

import math

import pytest

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r10e1 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_1


def test_working_model_adds_the_residual_to_the_mean_function():
    # (10.1): Z_k = m(x_k) + eps_k
    for m, eps in ((3.0, 0.5), (-1.0, 2.25), (0.0, 0.0)):
        assert the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_1(m, eps)["value"] == pytest.approx(m + eps, rel=1e-12)


def test_a_zero_residual_returns_the_mean_function():
    assert the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_1(7.5, 0.0)["value"] == pytest.approx(7.5, rel=1e-12)
