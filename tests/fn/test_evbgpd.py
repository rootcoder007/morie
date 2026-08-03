"""Tests for evbgpd.evt_bayes_gpd."""
from morie.fn.evbgpd import evt_bayes_gpd
from morie.fn.evgpds import evt_gpd_sample


def test_posterior_centres_on_truth():
    y = evt_gpd_sample(600, 1.5, 0.2, seed=10)["y"]
    r = evt_bayes_gpd(y, n_draws=1500, seed=2)
    sig = [d[0] for d in r["draws"]]
    xis = [d[1] for d in r["draws"]]
    assert abs(sum(sig) / len(sig) - 1.5) < 0.4
    assert abs(sum(xis) / len(xis) - 0.2) < 0.15
