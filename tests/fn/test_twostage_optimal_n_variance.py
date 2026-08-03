"""Tests for morie.fn.twostage_optimal_n_variance.

Brus, D. J. (2022). Spatial Sampling with R, eq. (7.9).
Inputs are chosen so the expected value is exact by hand:
  n = (S_w S_b sqrt(c2/c1) + S_b^2)/V_max
    = (8*4*sqrt(1/9) + 16)/2 = (32/3 + 16)/2
"""

import pytest

from morie.fn.twostage_optimal_n_variance import twostage_optimal_n_variance


def test_twostage_optimal_n_variance_matches_the_book_equation():
    r = twostage_optimal_n_variance(8.0, 4.0, 9.0, 1.0, 2.0)
    assert r["value"] == pytest.approx((32.0 / 3.0 + 16.0) / 2.0, abs=1e-12)


def test_twostage_optimal_n_variance_is_inverse_in_the_variance_target():
    a = twostage_optimal_n_variance(8.0, 4.0, 9.0, 1.0, 2.0)["value"]
    b = twostage_optimal_n_variance(8.0, 4.0, 9.0, 1.0, 4.0)["value"]
    assert b == pytest.approx(a / 2, abs=1e-12)


def test_twostage_optimal_n_variance_rejects_nonpositive_input():
    with pytest.raises(ValueError):
        twostage_optimal_n_variance(8.0, 4.0, 9.0, 1.0, 0.0)
