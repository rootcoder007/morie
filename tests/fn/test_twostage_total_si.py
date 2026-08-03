"""Tests for morie.fn.twostage_total_si.

Brus, D. J. (2022). Spatial Sampling with R, eq. (7.13).
Inputs are chosen so the expected value is exact by hand:
  t_hat = (N/n) sum t = (9/3) * (10+20+30) = 3 * 60 = 180
"""

import pytest

from morie.fn.twostage_total_si import twostage_total_si


def test_twostage_total_si_matches_the_book_equation():
    r = twostage_total_si([10.0, 20.0, 30.0], 9.0)
    assert r["value"] == pytest.approx(180.0, abs=1e-12)


def test_twostage_total_si_equals_the_sum_when_all_psus_are_sampled():
    # N = n means the sample is the population, so no scaling up occurs
    assert twostage_total_si([10.0, 20.0, 30.0], 3.0)["value"] == pytest.approx(
        60.0, abs=1e-12)


def test_twostage_total_si_rejects_bad_input():
    with pytest.raises(ValueError):
        twostage_total_si([10.0, 20.0], 0.0)
