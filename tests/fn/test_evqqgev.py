"""Tests for evqqgev.evt_gev_qq_plot."""
from morie.fn.evgevs import evt_gev_sample
from morie.fn.evqqgev import evt_gev_qq_plot


def test_line_under_true_model():
    x = evt_gev_sample(800, 5.0, 1.5, 0.1, seed=6)["x"]
    r = evt_gev_qq_plot(x, 5.0, 1.5, 0.1)
    # interior quantiles agree closely; tails wobble by construction
    mid = slice(80, 720)
    err = [abs(a - b) for a, b in
           zip(r["q_emp"][mid], r["q_model"][mid])]
    assert sum(err) / len(err) < 0.25
