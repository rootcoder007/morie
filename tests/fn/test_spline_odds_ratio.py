"""Tests for spline_odds_ratio.spline_odds_ratio (Bilder and Loughin eq. 6.37)."""

import math

import pytest

from morie.fn.spline_odds_ratio import spline_odds_ratio


BASIS = [lambda x: x, lambda x: max(x - 1.0, 0.0) ** 3, lambda x: max(x - 2.5, 0.0) ** 3]
BETAS = [0.4, -0.3, 0.25]


def _f(x):
    return sum(b * h(x) for b, h in zip(BETAS, BASIS))


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e37_basic():
    """OR(a, b) = exp(f(a) - f(b)) with f = sum beta_j h_j (eq. 6.36);
    odds ratios chain: OR(a, b) OR(b, c) = OR(a, c); below the first
    knot only the linear term contributes, OR = exp(beta_1 (a - b))."""
    r = spline_odds_ratio(BETAS, BASIS, 3.0, 0.5)
    assert r["value"] == pytest.approx(math.exp(_f(3.0) - _f(0.5)), rel=1e-15)
    ab = spline_odds_ratio(BETAS, BASIS, 3.0, 2.0)["value"]
    bc = spline_odds_ratio(BETAS, BASIS, 2.0, 0.5)["value"]
    assert ab * bc == pytest.approx(r["value"], rel=1e-14)
    assert spline_odds_ratio(BETAS, BASIS, 0.9, 0.2)["value"] == pytest.approx(math.exp(0.4 * 0.7), rel=1e-15)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e37_edge():
    """A coefficient vector that does not match the basis raises; equal
    points give an odds ratio of exactly one."""
    with pytest.raises(ValueError):
        spline_odds_ratio(BETAS[:2], BASIS, 1.0, 2.0)
    assert spline_odds_ratio(BETAS, BASIS, 1.7, 1.7)["value"] == 1.0
