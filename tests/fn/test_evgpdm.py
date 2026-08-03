"""Tests for evgpdm.evt_gpd_mle."""
from morie.fn.evgpdm import evt_gpd_mle
from morie.fn.evgpds import evt_gpd_sample


def test_parameter_recovery():
    y = evt_gpd_sample(1500, 1.5, 0.25, seed=7)["y"]
    f = evt_gpd_mle(y)
    assert abs(f["sigma"] - 1.5) < 0.25
    assert abs(f["xi"] - 0.25) < 0.1
