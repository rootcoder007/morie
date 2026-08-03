"""Tests for evgevm.evt_gev_mle."""
from morie.fn.evgevm import evt_gev_mle
from morie.fn.evgevs import evt_gev_sample


def test_parameter_recovery():
    x = evt_gev_sample(1500, 10.0, 2.0, 0.2, seed=42)["x"]
    f = evt_gev_mle(x)
    assert abs(f["mu"] - 10.0) < 0.3
    assert abs(f["sigma"] - 2.0) < 0.3
    assert abs(f["xi"] - 0.2) < 0.1
    assert f["cov"][0][0] > 0 and f["cov"][2][2] > 0
