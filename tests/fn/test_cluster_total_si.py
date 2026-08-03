"""Tests for morie.fn.cluster_total_si.

Brus, D. J. (2022). Spatial Sampling with R, eq. (6.9).
Inputs are chosen so the expected value is exact by hand:
  sum t = 12+18+9+21 = 60
  t_hat = (N/n) * 60 = (50/4) * 60 = 750
"""

import pytest

from morie.fn.cluster_total_si import cluster_total_si


def test_cluster_total_si_matches_the_book_equation():
    r = cluster_total_si([12.0, 18.0, 9.0, 21.0], 50.0, 4)
    assert r["value"] == pytest.approx(750.0, abs=1e-12)


def test_cluster_total_si_scales_linearly_with_the_population_count():
    a = cluster_total_si([2.0, 4.0], 10.0, 2)["value"]
    b = cluster_total_si([2.0, 4.0], 20.0, 2)["value"]
    assert b == pytest.approx(2 * a, abs=1e-12)


def test_cluster_total_si_rejects_bad_input():
    with pytest.raises(ValueError):
        cluster_total_si([1.0, 2.0], 10.0, 5)     # n /= sample size
    with pytest.raises(ValueError):
        cluster_total_si([1.0, 2.0], 0.0, 2)      # empty population
