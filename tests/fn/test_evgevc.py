"""Tests for evgevc.evt_gev_cdf."""
import math

from morie.fn.evgevc import evt_gev_cdf


def test_gumbel_at_location():
    # G(mu) = exp(-1) exactly in the Gumbel case (Coles eq. 3.2 limit)
    r = evt_gev_cdf(0.0, 0.0, 1.0, 0.0)
    assert abs(r["F"] - math.exp(-1.0)) < 1e-12


def test_support_edges():
    # xi > 0: mass 0 below mu - sigma/xi; xi < 0: mass 1 above bound
    assert evt_gev_cdf(-10.0, 0.0, 1.0, 0.5)["F"] == 0.0
    assert evt_gev_cdf(10.0, 0.0, 1.0, -0.5)["F"] == 1.0


def test_monotone_vector():
    r = evt_gev_cdf([0.0, 1.0, 2.0, 3.0], 1.0, 2.0, 0.2)
    F = r["F"]
    assert all(F[i] < F[i + 1] for i in range(3))
