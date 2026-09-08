# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for clustered DML + spatiotemporal Hawkes (R parity: rmorie)."""

from __future__ import annotations

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
import pytest

from morie import hawkes_spatial as hs
from morie.dml_clustered import dml_clustered


def _clustered(G=80, ng=15, seed=1):
    # Corridor-policy DGP: treatment assigned at the CORRIDOR level via observed
    # corridor covariate z; corridor random effect u in the outcome. True ATE=2.
    rng = np.random.default_rng(seed)
    n = G * ng
    g = np.repeat(np.arange(G), ng)
    z_g = rng.normal(size=G)
    d_g = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * z_g))))
    u_g = rng.normal(0, 1.0, G)
    z, d, u = z_g[g], d_g[g], u_g[g]
    x = rng.normal(size=n)
    y = 2 * d + 0.5 * z + x + u + rng.normal(0, 0.5, n)   # true ATE = 2
    return pd.DataFrame({"y": y, "d": d, "x": x, "z": z, "corridor": g})


def test_dml_clustered_recovers_ate():
    df = _clustered()
    res = dml_clustered(df, "d", "y", ["x", "z"], cluster="corridor")
    assert res["ate"] == pytest.approx(2, abs=0.5)
    assert res["se"] > 0
    assert res["n_clusters"] == 80


def test_cluster_se_exceeds_iid():
    df = _clustered()
    iid = dml_clustered(df, "d", "y", ["x", "z"], cluster=None)
    cl = dml_clustered(df, "d", "y", ["x", "z"], cluster="corridor")
    assert iid["se_kind"] == "iid"
    assert "cluster-robust" in cl["se_kind"]
    assert cl["se"] > iid["se"]


def test_two_way_cluster_runs():
    df = _clustered()
    df["week"] = np.tile(np.repeat(np.arange(6), 2), len(df) // 12 + 1)[: len(df)]
    res = dml_clustered(df, "d", "y", ["x", "z"], cluster=["corridor", "week"])
    assert "2-way" in res["se_kind"]
    assert np.isfinite(res["se"])


def test_dml_validation():
    df = _clustered()
    with pytest.raises(ValueError, match="at most two-way"):
        dml_clustered(df, "d", "y", ["x", "z"], cluster=["a", "b", "c"])
    with pytest.raises(ValueError, match="not found"):
        dml_clustered(df, "d", "y", ["nope"])


def _pars():
    return {"mu": 0.2, "alpha": 0.5, "beta": 1.0, "sigma": 0.4}


def test_intensity_background_and_decay():
    ev = pd.DataFrame({"t": [0.1, 0.5], "x": [0, 1], "y": [0, 1]})
    p = _pars()
    assert hs.hawkes_st_intensity(ev, -1, 0, 0, p) == pytest.approx(p["mu"])
    near = hs.hawkes_st_intensity(ev, 0.51, 0, 0, p)
    far = hs.hawkes_st_intensity(ev, 5.0, 0, 0, p)
    assert near > p["mu"] > 0
    assert far < near
    close = hs.hawkes_st_intensity(ev, 0.6, 0, 0, p)
    distant = hs.hawkes_st_intensity(ev, 0.6, 8, 8, p)
    assert close > distant


def test_loglik_peaks_near_truth():
    ev = hs.hawkes_st_simulate(_pars(), 40, (0, 10, 0, 10), seed=3)
    ll_true = hs.hawkes_st_loglik(ev, _pars(), end_time=40, area=100)
    assert np.isfinite(ll_true)
    bad = {**_pars(), "alpha": 0.95}
    assert ll_true > hs.hawkes_st_loglik(ev, bad, end_time=40, area=100)


def test_simulation_clusters_with_alpha():
    lo = hs.hawkes_st_simulate({"mu": 0.2, "alpha": 0.1, "beta": 1, "sigma": 0.4},
                               50, (0, 10, 0, 10), seed=11)
    hi = hs.hawkes_st_simulate({"mu": 0.2, "alpha": 0.7, "beta": 1, "sigma": 0.4},
                               50, (0, 10, 0, 10), seed=11)
    assert len(hi) > len(lo)
    assert (hi["t"] < 50).all()
    assert (hi["gen"] >= 0).all()
    assert (hi["gen"] > 0).any()


def test_simulation_rejects_supercritical():
    with pytest.raises(ValueError, match="subcritical"):
        hs.hawkes_st_simulate({"mu": 1, "alpha": 1.2, "beta": 1, "sigma": 1},
                              10, (0, 1, 0, 1))


def test_mle_maximises_likelihood_and_is_stable():
    truth = {"mu": 0.3, "alpha": 0.5, "beta": 1.2, "sigma": 0.5}
    ev = hs.hawkes_st_simulate(truth, 40, (0, 6, 0, 6), seed=5)
    ll_truth = hs.hawkes_st_loglik(ev, truth, end_time=40, area=36)
    fit = hs.hawkes_st_fit(ev, end_time=40, area=36)
    assert fit["convergence"] == 0
    # Weakly identified: assert likelihood maximisation + stability, not recovery.
    assert fit["loglik"] >= ll_truth - 1e-6
    assert 0 < fit["params"]["alpha"] < 1
    assert fit["params"]["mu"] > 0
    assert fit["params"]["beta"] > 0
    assert fit["params"]["sigma"] > 0


def test_param_validation():
    with pytest.raises(ValueError, match="mu, alpha, beta, sigma"):
        hs.hawkes_st_loglik(pd.DataFrame({"t": [1], "x": [0], "y": [0]}), {"mu": 0.1})
