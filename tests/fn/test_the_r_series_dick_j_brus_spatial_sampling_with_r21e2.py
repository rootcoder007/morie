"""Verification tests for the_r_series_dick_j_brus_spatial_sampling_with_r21e2.

Brus (2022), Spatial Sampling with R, eq. (21.2), the constant-mean model. Expected values are
recomputed from the formula in the test body.
"""

import math

import pytest

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r21e2 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_2


def test_constant_mean_model_adds_the_residual():
    # (21.2): Z(s) = mu + eps(s)
    for mu, eps in ((3.0, 0.25), (-2.0, 1.5), (0.0, 0.0)):
        assert the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_2(mu, eps)["value"] == pytest.approx(mu + eps, rel=1e-12)


def test_the_expected_value_is_the_mean_when_the_residual_vanishes():
    assert the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_2(4.25, 0.0)["value"] == pytest.approx(4.25, rel=1e-12)
