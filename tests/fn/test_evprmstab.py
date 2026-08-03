"""Tests for evprmstab.evt_param_stability."""
from morie.fn.evgpds import evt_gpd_sample
from morie.fn.evprmstab import evt_param_stability


def test_sigma_star_stable_for_pure_gpd():
    # exact GPD data: sigma* = sigma_u - xi*u is constant in u
    # (Coles 2001 eq. 4.9), up to sampling noise
    y = evt_gpd_sample(4000, 2.0, 0.2, seed=7)["y"]
    r = evt_param_stability(y)
    ss = r["sigma_star"]
    assert len(ss) >= 5
    assert max(ss) - min(ss) < 1.0
