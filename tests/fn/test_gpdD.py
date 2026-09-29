"""Tests for morie.fn.gpdD: GPD functions recomputed from the closed forms."""

import math

from morie.fn.gpdD import gpd_distribution


def test_cdf_pdf_quantile_moments():
    s, k = 2.0, 0.25
    r = gpd_distribution(s, k, x=[1.0, 3.0], p=[0.5, 0.9])
    for x, c, d in zip([1.0, 3.0], r["cdf"], r["pdf"]):
        z = 1 + k * x / s
        assert abs(c - (1 - z ** (-1 / k))) < 1e-15
        assert abs(d - z ** (-1 / k - 1) / s) < 1e-15
    for p, q in zip([0.5, 0.9], r["quantile"]):
        assert abs(1 - (1 + k * q / s) ** (-1 / k) - p) < 1e-14
    assert abs(r["mean"] - s / (1 - k)) < 1e-15
    assert abs(r["variance"] - s * s / ((1 - k) ** 2 * (1 - 2 * k))) < 1e-14


def test_exponential_limit_and_bounded_support():
    r = gpd_distribution(1.5, 0.0, x=[2.0], p=[0.75])
    assert abs(r["cdf"][0] - (1 - math.exp(-2 / 1.5))) < 1e-15
    assert abs(r["quantile"][0] + 1.5 * math.log(0.25)) < 1e-15
    assert gpd_distribution(1.0, -0.5)["upper_endpoint"] == 2.0
