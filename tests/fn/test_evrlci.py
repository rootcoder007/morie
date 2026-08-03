"""Tests for evrlci.evt_return_level_ci."""
from morie.fn.evgevs import evt_gev_sample
from morie.fn.evrlci import evt_return_level_ci


def test_ci_brackets_truth():
    from morie.fn._evt_core import gev_return_level
    x = evt_gev_sample(1500, 10.0, 2.0, 0.1, seed=3)["x"]
    r = evt_return_level_ci(x, 50)
    true_z = gev_return_level(50, 10.0, 2.0, 0.1)
    assert r["ci_lo"] < r["z_T"] < r["ci_hi"]
    assert r["ci_lo"] < true_z < r["ci_hi"]
