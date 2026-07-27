"""IV/bounds/refutation cluster: mivbnd, causdmliv, ddrbnd, gb1251,
plcbo, causftbl, clstcr, shrtgr, causshap, abdpd, counRS, bnscrd,
causrho, msmiv2, msmphr, fciag, deciA."""

import numpy as np
import pytest

from morie.fn.abdpd import abduction_modification_prediction
from morie.fn.bnscrd import bound_causal_rd
from morie.fn.causdmliv import causal_dml_iv
from morie.fn.causftbl import causal_frontdoor_adjustment
from morie.fn.causrho import causal_proximal_proxy
from morie.fn.causshap import causal_shap_decomposition
from morie.fn.clstcr import cluster_causal_inference
from morie.fn.counRS import counterfactual_rec
from morie.fn.ddrbnd import deer_dr_bounds
from morie.fn.deciA import deci_model
from morie.fn.fciag import fci_algorithm
from morie.fn.gb1251 import gibbons_partial_tau
from morie.fn.mivbnd import monotone_iv_bounds
from morie.fn.msmiv2 import msm_iv
from morie.fn.msmphr import msm_proportional_hazards
from morie.fn.plcbo import placebo_refutation
from morie.fn.shrtgr import shrinkage_propensity


def test_mivbnd_contains_truth_and_sharpens():
    rng = np.random.default_rng(0)
    n = 4000
    z = rng.integers(0, 4, n).astype(float)
    d = (rng.random(n) < 0.2 + 0.2 * z).astype(float)
    y = np.clip(0.3 * z + 2.0 * d + rng.normal(scale=0.3, size=n), -3, 6)
    out = monotone_iv_bounds(y, d, z, y_min=-3, y_max=6)
    assert out["lower"] <= 2.0 <= out["upper"]  # bounds cover the truth
    w_lo, w_hi = out["worst_case"]
    assert out["width"] <= (w_hi - w_lo) + 1e-9  # MIV never wider than Manski
    with pytest.raises(ValueError):
        monotone_iv_bounds(y, d, np.zeros(n))  # single instrument level


def test_causdmliv_recovers_theta_under_confounding():
    hits = 0
    for seed in range(6):
        rng = np.random.default_rng(seed)
        n = 3000
        X = rng.normal(size=(n, 4))
        u = rng.normal(size=n)  # unobserved confounder
        z = rng.normal(size=n)
        d = 0.8 * z + u + X @ np.full(4, 0.3) + rng.normal(scale=0.5, size=n)
        y = 1.5 * d + 2.0 * u + X @ np.full(4, -0.2) + rng.normal(scale=0.5, size=n)
        out = causal_dml_iv(y, d, z, X, n_folds=5, seed=seed)
        ols = np.linalg.lstsq(np.column_stack([np.ones(n), d, X]), y, rcond=None)[0][1]
        assert abs(ols - 1.5) > 0.3  # OLS badly biased by u
        hits += abs(out["theta"] - 1.5) < 0.2
    assert hits >= 5  # measured 6/6


def test_ddrbnd_late_recovery():
    hits = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n = 4000
        x = rng.normal(size=n)
        z = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(float)
        # compliers ~ 60%, always-takers 20%; effect 2 among compliers
        typ = rng.choice(["c", "a", "n"], size=n, p=[0.6, 0.2, 0.2])
        d = np.where(typ == "a", 1.0, np.where(typ == "n", 0.0, z))
        y = 2.0 * d * (typ == "c") + 0.5 * x + rng.normal(scale=0.5, size=n)
        out = deer_dr_bounds(y, d, z, x)
        assert out["defier_check"] is True
        assert 0.4 < out["compliance"] < 0.8
        hits += abs(out["late"] - 2.0) < 0.3
    assert hits >= 7  # measured 8/8


def test_gb1251_partial_tau_kills_confounded_association():
    rng = np.random.default_rng(0)
    n = 2000
    z = rng.normal(size=n)
    x = z + rng.normal(scale=0.4, size=n)
    y = z + rng.normal(scale=0.4, size=n)  # x-y association is all through z
    out = gibbons_partial_tau(x, y, z)
    assert out["tau_xy"] > 0.4
    # partial tau shrinks the association sharply but, unlike the
    # Gaussian partial correlation, does not go to zero under exact
    # conditional independence (measured 0.199 vs a marginal 0.60)
    assert abs(out["partial_tau"]) < 0.5 * out["tau_xy"]
    with pytest.raises(ValueError):
        gibbons_partial_tau([1.0, 2.0], [1.0, 2.0], [1.0, 2.0])  # n < 4


def test_plcbo_rejects_real_and_passes_placebo():
    rng = np.random.default_rng(0)
    n = 400
    t = (rng.random(n) < 0.5).astype(float)
    y_eff = 2.0 * t + rng.normal(size=n)
    y_null = rng.normal(size=n)

    def diff(y, tr):
        return float(y[tr == 1].mean() - y[tr == 0].mean())

    real = placebo_refutation(diff, y_eff, t, n_simulations=200, seed=0)
    assert real["p_value"] < 0.05 and real["passes"] is True
    assert abs(real["placebo_mean"]) < 0.3
    null = placebo_refutation(diff, y_null, t, n_simulations=200, seed=0)
    assert null["p_value"] > 0.05
    with pytest.raises(ValueError):
        placebo_refutation("not callable", y_eff, t)


def test_causftbl_table_frontdoor():
    # X -> Z -> Y with X and Y confounded; tables chosen so do(X=1) is computable
    P_Z_X = np.array([[0.9, 0.1], [0.2, 0.8]])
    P_Y_XZ = np.array([[[0.8, 0.2], [0.3, 0.7]], [[0.6, 0.4], [0.1, 0.9]]])
    P_X = np.array([0.5, 0.5])
    out = causal_frontdoor_adjustment(P_Z_X, P_Y_XZ, P_X)
    assert out["p_y_do_x"].shape == (2, 2)
    assert out["p_y_do_x"].sum(axis=1) == pytest.approx([1.0, 1.0])
    # hand check for x = 0: sum_z P(z|0) sum_x' P(y=1|x',z) P(x')
    inner1 = np.array([0.5 * 0.2 + 0.5 * 0.4, 0.5 * 0.7 + 0.5 * 0.9])
    assert out["p_y_do_x"][0, 1] == pytest.approx(0.9 * inner1[0] + 0.1 * inner1[1])
    with pytest.raises(ValueError):
        causal_frontdoor_adjustment(np.array([[0.5, 0.9]]), P_Y_XZ[:1], [1.0])


def test_clstcr_cluster_se_exceeds_naive():
    rng = np.random.default_rng(0)
    G, npc = 40, 50
    n = G * npc
    cl = np.repeat(np.arange(G), npc)
    D = np.repeat((rng.random(G) < 0.5).astype(float), npc)
    y = 1.0 * D + np.repeat(rng.normal(scale=1.5, size=G), npc) + rng.normal(scale=0.5, size=n)
    out = cluster_causal_inference(y, D, cl)
    assert out["se_cluster"] > 3 * out["se_naive"]  # measured ~10x with strong ICC
    assert out["estimate"] == pytest.approx(1.0, abs=3 * out["se_cluster"])
    assert out["icc"] > 0.5
    assert out["n_clusters"] == G
    with pytest.raises(ValueError):
        cluster_causal_inference(y, D, np.zeros(n))  # 1 cluster


def test_shrtgr_shrinks_towards_the_interior():
    # separable data: unpenalised logistic diverges, shrinkage does not
    rng = np.random.default_rng(0)
    n = 200
    h = rng.normal(size=n)
    A = (h > 0).astype(float)  # perfectly separated
    tight = shrinkage_propensity(A, h, prior_sd=0.5)
    loose = shrinkage_propensity(A, h, prior_sd=100.0)
    assert tight["ps_min"] > loose["ps_min"]
    assert tight["ps_max"] < loose["ps_max"]
    assert abs(tight["coefficients"][1]) < abs(loose["coefficients"][1])
    with pytest.raises(ValueError):
        shrinkage_propensity(A, h, prior_sd=0.0)


def test_causshap_efficiency_and_symmetry():
    # additive value function: each player's Shapley value is its own term
    w = {"a": 3.0, "b": 2.0, "c": -1.0}

    def v(S):
        return sum(w[f] for f in S)

    out = causal_shap_decomposition(v, ["a", "b", "c"])
    assert out["shapley"] == pytest.approx(w)
    assert out["efficiency_gap"] == pytest.approx(0.0, abs=1e-12)
    # symmetric players (interaction only) split the surplus equally
    def v2(S):
        return 1.0 if set(S) == {"a", "b"} else 0.0

    sym = causal_shap_decomposition(v2, ["a", "b"])
    assert sym["shapley"]["a"] == pytest.approx(0.5)
    assert sym["shapley"]["b"] == pytest.approx(0.5)
    samp = causal_shap_decomposition(v, ["a", "b", "c"], n_samples=400, seed=0)
    assert samp["shapley"]["a"] == pytest.approx(3.0, abs=1e-9)  # additive: exact per draw
    assert samp["exact"] is False


def test_abdpd_three_steps():
    # Y = 2X + u2, X = u1. Observe X = 1, Y = 3 -> u2 = 1.
    eqs = {"X": (("u1",), lambda u1: u1), "Y": (("X", "u2"), lambda X, u2: 2 * X + u2)}
    out = abduction_modification_prediction(
        {"X": 1.0, "Y": 3.0}, eqs, ["u1", "u2"], {"X": 4.0}, "Y"
    )
    assert out["abducted"]["u1"] == pytest.approx(1.0, abs=1e-6)
    assert out["abducted"]["u2"] == pytest.approx(1.0, abs=1e-6)
    assert out["factual"] == pytest.approx(3.0, abs=1e-6)
    assert out["counterfactual"] == pytest.approx(9.0, abs=1e-6)  # 2*4 + 1
    with pytest.raises(ValueError):
        abduction_modification_prediction({"X": 1.0}, eqs, ["u1", "u2"], {"u1": 0.0}, "Y")


def test_counRS_offpolicy_estimators():
    rng = np.random.default_rng(0)
    n, k = 4000, 3
    p0 = np.full((n, k), 1 / k)
    a = rng.integers(0, k, n)
    true_r = np.array([0.1, 0.5, 0.9])
    r = true_r[a] + rng.normal(scale=0.1, size=n)
    # target policy always plays action 2 -> true value 0.9
    target = np.zeros((n, k))
    target[:, 2] = 1.0
    out = counterfactual_rec(a, r, p0[np.arange(n), a], target)
    assert out["ips"] == pytest.approx(0.9, abs=0.05)
    assert out["snips"] == pytest.approx(0.9, abs=0.05)
    assert out["dr"] is None
    q = np.tile(true_r, (n, 1))
    dr = counterfactual_rec(a, r, p0[np.arange(n), a], target, reward_model=q)
    assert dr["dr"] == pytest.approx(0.9, abs=0.03)
    assert dr["ess"] < n  # reweighting costs sample size
    clipped = counterfactual_rec(a, r, p0[np.arange(n), a], target, clip=1.0)
    assert clipped["n_clipped"] > 0
    with pytest.raises(ValueError):
        counterfactual_rec(a, r, np.zeros(n), target)  # zero logging probability


def test_bnscrd_bounds_bracket_the_estimate():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.uniform(-1, 1, n)
    y = 1.0 * (x >= 0) + 0.5 * x + rng.normal(scale=0.2, size=n)
    obs = rng.random(n) > 0.15  # 15% missing
    out = bound_causal_rd(y, x, 0.0, observed=obs, bandwidth=0.5, y_min=-2, y_max=3)
    assert out["lower"] <= out["estimate"] <= out["upper"]
    assert out["width"] > 0
    assert out["n_missing"] > 0
    full = bound_causal_rd(y, x, 0.0, bandwidth=0.5)
    assert full["width"] == pytest.approx(0.0, abs=1e-9)  # nothing missing
    assert full["estimate"] == pytest.approx(1.0, abs=0.15)


def test_causrho_proximal_beats_naive():
    hits = 0
    for seed in range(6):
        rng = np.random.default_rng(seed)
        n = 5000
        u = rng.normal(size=n)
        z = u + rng.normal(scale=0.5, size=n)  # treatment-inducing proxy
        w = u + rng.normal(scale=0.5, size=n)  # outcome-inducing proxy
        a = 0.8 * u + rng.normal(scale=0.6, size=n)
        y = 1.0 * a + 1.5 * u + rng.normal(scale=0.5, size=n)
        out = causal_proximal_proxy(y, a, z, w)
        hits += abs(out["estimate"] - 1.0) < abs(out["naive"] - 1.0)
        assert out["first_stage_r2"][0] > 0.2
    assert hits == 6  # measured 6/6
    with pytest.raises(ValueError):
        causal_proximal_proxy(np.zeros(10), np.zeros(10), np.zeros((10, 1)), np.zeros((10, 2)))


def test_msmiv2_2sls_and_weak_flag():
    rng = np.random.default_rng(0)
    n = 3000
    u = rng.normal(size=n)
    z = rng.normal(size=n)
    a1 = 0.7 * z + u + rng.normal(scale=0.5, size=n)
    y = 1.2 * a1 + 2.0 * u + rng.normal(scale=0.5, size=n)
    out = msm_iv(y, a1, z)
    assert out["estimate"] == pytest.approx(1.2, abs=0.15)
    assert abs(out["ols_estimate"] - 1.2) > 0.3  # confounded
    assert out["weak_instrument"] is False
    weak = msm_iv(y, a1, rng.normal(size=n) * 0.001)
    assert weak["weak_instrument"] is True


def test_msmphr_marginal_hazard_ratio():
    hits = 0
    for seed in range(6):
        rng = np.random.default_rng(seed)
        n = 4000
        c = rng.normal(size=n)
        A = (rng.random(n) < 1 / (1 + np.exp(-1.2 * c))).astype(float)
        t_event = rng.exponential(np.exp(-(0.5 * A + 0.8 * c)))
        cens = rng.exponential(3.0, size=n)
        time = np.minimum(t_event, cens)
        event = (t_event <= cens).astype(float)
        out = msm_proportional_hazards(time, event, A, c)
        # weighting moves the estimate toward the true marginal effect
        hits += abs(out["log_hr"] - 0.5) < abs(out["log_hr_unweighted"] - 0.5)
        assert out["hazard_ratio"] == pytest.approx(np.exp(out["log_hr"]))
    assert hits >= 5  # measured 6/6


def test_fciag_recovers_skeleton_and_collider():
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    cc = a + b + rng.normal(scale=0.4, size=n)  # collider A -> C <- B
    d = cc + rng.normal(scale=0.4, size=n)
    out = fci_algorithm(np.column_stack([a, b, cc, d]), names=["A", "B", "C", "D"])
    edges = {frozenset(e) for e in out["edges"]}
    assert frozenset({"A", "C"}) in edges
    assert frozenset({"B", "C"}) in edges
    assert frozenset({"C", "D"}) in edges
    assert frozenset({"A", "B"}) not in edges  # marginally independent
    assert ("A", "C", "B") in out["colliders"] or ("B", "C", "A") in out["colliders"]
    assert out["orientation_complete"] is False
    with pytest.raises(ValueError):
        fci_algorithm(np.zeros((10, 2)))


def test_deciA_pipeline_adjusts():
    rng = np.random.default_rng(0)
    n = 4000
    conf = rng.normal(size=n)
    t = 0.9 * conf + rng.normal(scale=0.6, size=n)
    y = 1.0 * t + 1.5 * conf + rng.normal(scale=0.5, size=n)
    data = np.column_stack([conf, t, y])
    out = deci_model(data, "T", "Y", names=["U", "T", "Y"])
    assert "U" in out["adjustment_set"]
    assert out["estimate"] == pytest.approx(1.0, abs=0.1)
    assert abs(out["naive"] - 1.0) > 0.3
    dag = {"U": ["T", "Y"], "T": ["Y"]}
    ver = deci_model(data, "T", "Y", names=["U", "T", "Y"], dag=dag)
    assert ver["backdoor_verified"] is True
    with pytest.raises(ValueError):
        deci_model(data, "T", "T", names=["U", "T", "Y"])
