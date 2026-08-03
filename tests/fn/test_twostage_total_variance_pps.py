"""Tests for morie.fn.twostage_total_variance_pps.

Brus, D. J. (2022). Spatial Sampling with R, eq. (7.12).
Inputs are chosen so the expected value is exact by hand:
  first  = sum p (t/p - t_total)^2 / n
         = (225 + 25 + 50)/2 = 150
  second = sum M^2 (1-f2) S2 / (m p) / n = (32 + 32 + 32)/2 = 48
  V      = 198
"""

import pytest

from morie.fn.twostage_total_variance_pps import twostage_total_variance_pps


def test_twostage_total_variance_pps_matches_the_book_equation():
    r = twostage_total_variance_pps(
        [0.25, 0.25, 0.5], [10.0, 20.0, 40.0], 70.0,
        [4.0, 4.0, 8.0], [0.5, 0.5, 0.5], [2.0, 2.0, 2.0],
        [2.0, 2.0, 4.0], 2)
    assert r["value"] == pytest.approx(198.0, abs=1e-12)


def test_twostage_total_variance_pps_second_term_vanishes_at_full_second_stage():
    # f2 = 1 means every unit in the PSU was measured, so the
    # within-PSU term contributes nothing and only the between term is left
    r = twostage_total_variance_pps(
        [0.25, 0.25, 0.5], [10.0, 20.0, 40.0], 70.0,
        [4.0, 4.0, 8.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0],
        [2.0, 2.0, 4.0], 2)
    assert r["value"] == pytest.approx(150.0, abs=1e-12)


def test_twostage_total_variance_pps_rejects_bad_input():
    with pytest.raises(ValueError):
        twostage_total_variance_pps(
            [0.0, 0.5, 0.5], [10.0, 20.0, 40.0], 70.0,
            [4.0, 4.0, 8.0], [0.5, 0.5, 0.5], [2.0, 2.0, 2.0],
            [2.0, 2.0, 4.0], 2)
