"""Tests for morie.fn.cluster_total_pps.

Brus, D. J. (2022). Spatial Sampling with R, eq. (6.4).
Inputs are chosen so the expected value is exact by hand:
  t/M = 12/4 = 18/6 = 9/3 = 21/7 = 3, so sum = 12
  t_hat = (M/n) * 12 = (100/4) * 12 = 300
"""

import pytest

from morie.fn.cluster_total_pps import cluster_total_pps


def test_cluster_total_pps_matches_the_book_equation():
    r = cluster_total_pps([12.0, 18.0, 9.0, 21.0], [4.0, 6.0, 3.0, 7.0], 100.0, 4)
    assert r["value"] == pytest.approx(300.0, abs=1e-12)


def test_cluster_total_pps_is_the_scaled_sum_of_cluster_means():
    # every cluster mean is 3, so the estimator must be M * 3
    r = cluster_total_pps([6.0, 9.0], [2.0, 3.0], 50.0, 2)
    assert r["value"] == pytest.approx(50.0 / 2 * 6.0, abs=1e-12)


def test_cluster_total_pps_rejects_bad_input():
    with pytest.raises(ValueError):
        cluster_total_pps([1.0, 2.0], [1.0, 0.0], 10.0, 2)   # zero size
    with pytest.raises(ValueError):
        cluster_total_pps([1.0, 2.0], [1.0, 2.0], 10.0, 3)   # n /= sample
