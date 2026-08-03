"""Tests for evstabint.evt_xi_ci_profile."""
from morie.fn.evgevs import evt_gev_sample
from morie.fn.evstabint import evt_xi_ci_profile


def test_profile_interval_brackets_truth():
    x = evt_gev_sample(800, 10.0, 2.0, 0.15, seed=8)["x"]
    r = evt_xi_ci_profile(x)
    assert r["ci_lo"] < 0.15 < r["ci_hi"]
    assert r["ci_lo"] < r["xi_hat"] < r["ci_hi"]
