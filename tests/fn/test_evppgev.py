"""Tests for evppgev.evt_gev_pp_plot."""
from morie.fn.evgevs import evt_gev_sample
from morie.fn.evppgev import evt_gev_pp_plot


def test_diagonal_under_true_model():
    x = evt_gev_sample(800, 5.0, 1.5, 0.1, seed=5)["x"]
    r = evt_gev_pp_plot(x, 5.0, 1.5, 0.1)
    d = max(abs(a - b) for a, b in zip(r["p_emp"], r["p_model"]))
    assert d < 0.08          # KS-scale deviation for n = 800
