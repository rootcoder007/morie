"""Tests for evbgrev.evt_bayes_gev."""
from morie.fn.evbgrev import evt_bayes_gev
from morie.fn.evgevs import evt_gev_sample


def test_posterior_centres_on_truth():
    x = evt_gev_sample(600, 10.0, 2.0, 0.1, seed=9)["x"]
    r = evt_bayes_gev(x, n_draws=1500, seed=1)
    mus = [d[0] for d in r["draws"]]
    xis = [d[2] for d in r["draws"]]
    assert abs(sum(mus) / len(mus) - 10.0) < 0.5
    assert abs(sum(xis) / len(xis) - 0.1) < 0.15
    assert 0.1 < r["accept_rate"] < 0.8
