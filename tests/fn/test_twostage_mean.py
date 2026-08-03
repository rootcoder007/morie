"""Tests for morie.fn.twostage_mean.

Brus, D. J. (2022). Spatial Sampling with R, eq. (7.2).
Inputs are chosen so the expected value is exact by hand:
  zbarbar = (2+4+6+8)/4 = 5
"""

import pytest

from morie.fn.twostage_mean import twostage_mean


def test_twostage_mean_matches_the_book_equation():
    r = twostage_mean([2.0, 4.0, 6.0, 8.0])
    assert r["value"] == pytest.approx(5.0, abs=1e-12)


def test_twostage_mean_is_unweighted_across_primary_units():
    # eq (7.2) averages the PSU means, ignoring how many units each holds
    assert twostage_mean([0.0, 10.0])["value"] == pytest.approx(5.0, abs=1e-12)
