"""Tests for evrl.evt_return_level."""
from morie.fn.evgevc import evt_gev_cdf
from morie.fn.evrl import evt_return_level


def test_solves_return_period_equation():
    # z_T is the level with exceedance probability exactly 1/T
    for T in (10, 100, 1000):
        z = evt_return_level(10.0, 2.0, 0.2, T)["z_T"]
        F = evt_gev_cdf(z, 10.0, 2.0, 0.2)["F"]
        assert abs((1.0 - F) - 1.0 / T) < 1e-10
