"""Tests for evgofa -- Anderson-Darling A^2 for a fitted GEV.

Anchored on the xi = 0 limit, where Coles (2001) eq (3.2) collapses to
the Gumbel cdf exp(-exp(-(x-mu)/sigma)), written out by hand here.
"""

import math

import pytest

from morie.fn.evgofa import evt_gev_anderson_darling, gev_cdf
from morie.fn.hedderich7e33 import ad_statistic

X = [0.31, 0.87, 1.02, 1.44, 1.91, 2.05, 2.63, 3.10, 3.88, 5.02, 6.41, 8.20]


def test_zero_shape_is_the_gumbel():
    for x in (-1.0, 0.0, 1.0, 2.5, 7.0):
        assert float(gev_cdf([x], 1.0, 1.5, 0.0)[0]) == pytest.approx(
            math.exp(-math.exp(-(x - 1.0) / 1.5)), abs=1e-15
        )


def test_cdf_is_monotone_and_in_the_unit_interval():
    for xi in (-0.3, 0.0, 0.2):
        vals = [float(gev_cdf([x], 1.0, 1.5, xi)[0]) for x in [-2, -1, 0, 1, 2, 4, 8, 16]]
        assert all(0.0 <= v <= 1.0 for v in vals)
        assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_statistic_is_the_shared_ad_core_on_the_pit():
    res = evt_gev_anderson_darling(X, 1.0, 1.5, 0.2)
    assert res["statistic"] == pytest.approx(ad_statistic(gev_cdf(X, 1.0, 1.5, 0.2)))
    assert res["n"] == len(X)


def test_a_badly_wrong_location_fits_worse_than_a_good_one():
    good = evt_gev_anderson_darling(X, 1.0, 1.5, 0.2)["statistic"]
    bad = evt_gev_anderson_darling(X, 40.0, 1.5, 0.2)["statistic"]
    assert bad > good


def test_support_endpoint_is_respected():
    # xi < 0 gives a finite upper endpoint at mu - sigma/xi
    assert float(gev_cdf([1000.0], 0.0, 1.0, -0.5)[0]) == 1.0
    # xi > 0 gives a finite lower endpoint
    assert float(gev_cdf([-1000.0], 0.0, 1.0, 0.5)[0]) == 0.0


def test_rejects_nonpositive_scale():
    with pytest.raises(ValueError):
        gev_cdf([1.0], 0.0, 0.0, 0.0)
