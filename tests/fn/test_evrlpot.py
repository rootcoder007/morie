"""Tests for evrlpot.evt_return_level_pot."""
from morie.fn.evrlpot import evt_return_level_pot


def test_pot_level_above_threshold_and_monotone():
    z1 = evt_return_level_pot(5.0, 1.5, 0.2, 0.05, 365)["z_T"]
    z2 = evt_return_level_pot(5.0, 1.5, 0.2, 0.05, 3650)["z_T"]
    assert z1 > 5.0 and z2 > z1


def test_matches_gpd_quantile_identity():
    # x_m solves zeta_u * (1 - H(x_m - u)) = 1/m  (Coles eq. 4.12)
    from morie.fn._evt_core import gpd_cdf
    u, s, xi, zeta, m = 5.0, 1.5, 0.2, 0.05, 365
    z = evt_return_level_pot(u, s, xi, zeta, m)["z_T"]
    assert abs(zeta * (1.0 - gpd_cdf(z - u, s, xi)) - 1.0 / m) < 1e-12
