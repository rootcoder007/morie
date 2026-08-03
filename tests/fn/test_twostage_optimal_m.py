"""Tests for morie.fn.twostage_optimal_m.

Brus, D. J. (2022). Spatial Sampling with R, eq. (7.10).
Inputs are chosen so the expected value is exact by hand:
  m = (S_w/S_b) sqrt(c1/c2) = (8/4) * sqrt(9/1) = 2 * 3 = 6
"""

import pytest

from morie.fn.twostage_optimal_m import twostage_optimal_m


def test_twostage_optimal_m_matches_the_book_equation():
    r = twostage_optimal_m(8.0, 4.0, 9.0, 1.0)
    assert r["value"] == pytest.approx(6.0, abs=1e-12)


def test_twostage_optimal_m_grows_with_within_psu_variance():
    # more variation inside a PSU means sample more units within it
    small = twostage_optimal_m(4.0, 4.0, 9.0, 1.0)["value"]
    large = twostage_optimal_m(8.0, 4.0, 9.0, 1.0)["value"]
    assert large > small


def test_twostage_optimal_m_rejects_nonpositive_input():
    with pytest.raises(ValueError):
        twostage_optimal_m(0.0, 4.0, 9.0, 1.0)
