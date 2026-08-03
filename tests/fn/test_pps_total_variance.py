"""Tests for morie.fn.pps_total_variance.

Brus, D. J. (2022). Spatial Sampling with R, eq. (8.2).
Inputs are chosen so the expected value is exact by hand:
  z/p = 10 for every unit; (10-8)^2 = 4 three times, sum = 12
  V = 12/(n(n-1)) = 12/(3*2) = 2
"""

import pytest

from morie.fn.pps_total_variance import pps_total_variance


def test_pps_total_variance_matches_the_book_equation():
    r = pps_total_variance([2.0, 4.0, 6.0], [0.2, 0.4, 0.6], 8.0)
    assert r["value"] == pytest.approx(2.0, abs=1e-12)


def test_pps_total_variance_is_zero_when_every_ratio_hits_the_estimate():
    # z_k/p_k identical and equal to t_hat leaves nothing to vary
    r = pps_total_variance([2.0, 4.0, 6.0], [0.2, 0.4, 0.6], 10.0)
    assert r["value"] == pytest.approx(0.0, abs=1e-12)


def test_pps_total_variance_rejects_bad_input():
    with pytest.raises(ValueError):
        pps_total_variance([2.0], [0.2], 10.0)              # n < 2
    with pytest.raises(ValueError):
        pps_total_variance([2.0, 4.0], [0.0, 0.4], 10.0)    # zero probability
