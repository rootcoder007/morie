"""Tests for smt.semiparametric_max."""
from morie.fn.evgevs import evt_gev_sample
from morie.fn.smt import semiparametric_max


def test_recovers_linear_trend():
    base = evt_gev_sample(300, 5.0, 1.0, 0.1, seed=11)["x"]
    x = [v + 0.02 * i for i, v in enumerate(base)]
    r = semiparametric_max(x)
    assert abs(r["estimate"] - 0.02) < 0.008
    assert r["lr_vs_stationary"] > 3.84   # significant at 5%


def test_no_trend_small_beta():
    x = evt_gev_sample(300, 5.0, 1.0, 0.1, seed=12)["x"]
    r = semiparametric_max(x)
    assert abs(r["estimate"]) < 0.01
