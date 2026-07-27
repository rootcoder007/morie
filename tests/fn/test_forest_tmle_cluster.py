"""Forest + TMLE tier: crfath, crfboot, crfhte, csfgrf, survcfg, csurv2,
qbcfgr, htgcrf, drlnr, ipsiMed, tmlpoy, tmltrt, tmlsen, tmlqct, tmlmed,
tmlivc, tmltvc, tmllng, npstm."""

import numpy as np
import pytest

from morie.fn._tmle import tmle_ate
from morie.fn.crfath import causal_forest_wager_athey
from morie.fn.crfboot import causal_forest_bootstrap
from morie.fn.crfhte import causal_forest_hte_test
from morie.fn.csfgrf import causal_survival_forest
from morie.fn.csurv2 import causal_survival_blp
from morie.fn.drlnr import dr_learner
from morie.fn.htgcrf import hetero_causal_forest
from morie.fn.ipsiMed import interventional_psi
from morie.fn.npstm import nonparametric_tmle_survival
from morie.fn.qbcfgr import quantile_balanced_cf
from morie.fn.survcfg import causal_survival_forest_grf
from morie.fn.tmlivc import tmle_iv
from morie.fn.tmllng import tmle_longitudinal
from morie.fn.tmlmed import tmle_mediation
from morie.fn.tmlpoy import tmle_propensity_only
from morie.fn.tmlqct import tmle_quantile
from morie.fn.tmlsen import tmle_sensitivity_unmeasured
from morie.fn.tmltrt import tmle_truncation
from morie.fn.tmltvc import tmle_time_varying_confound


def _hetero(seed, n=1200):
    """tau(x) = 1 + 2*x0: heterogeneity in the first covariate only."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    D = (rng.random(n) < 0.5).astype(float)
    tau = 1.0 + 2.0 * X[:, 0]
    y = X[:, 1] + tau * D + rng.normal(scale=0.5, size=n)
    return y, D, X, tau


def _confounded(seed, n=2000):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n, 3))
    e = 1 / (1 + np.exp(-(W @ np.array([1.0, -0.5, 0.3]))))
    A = (rng.random(n) < e).astype(float)
    y = 2.0 * A + W @ np.full(3, 1.0) + rng.normal(scale=0.5, size=n)
    return y, A, W


def test_crfath_tracks_true_heterogeneity():
    hits = 0
    for seed in range(4):
        y, D, X, tau = _hetero(seed)
        out = causal_forest_wager_athey(y, D, X, n_trees=120, min_leaf=15, seed=seed)
        ok = np.isfinite(out["cate_oob"])
        r = np.corrcoef(out["cate_oob"][ok], tau[ok])[0, 1]
        hits += r > 0.5  # measured 0.62-0.75
        assert abs(out["ate"] - 1.0) < 0.5  # mean tau is 1
        assert out["cate_sd"] > 0.3  # a constant-effect forest would be flat
    assert hits >= 3


def test_crfhte_detects_and_rejects():
    y, D, X, tau = _hetero(0)
    f = causal_forest_wager_athey(y, D, X, n_trees=120, min_leaf=15, seed=0)
    het = causal_forest_hte_test(y, D, f["cate_oob"])
    assert het["heterogeneous"] is True
    assert het["beta"] > 0
    # constant-effect data: the same pipeline must not claim heterogeneity
    rng = np.random.default_rng(1)
    n = 1200
    Xc = rng.normal(size=(n, 3))
    Dc = (rng.random(n) < 0.5).astype(float)
    yc = Xc[:, 1] + 1.0 * Dc + rng.normal(scale=0.5, size=n)
    fc = causal_forest_wager_athey(yc, Dc, Xc, n_trees=120, min_leaf=15, seed=1)
    assert causal_forest_hte_test(yc, Dc, fc["cate_oob"])["p_value"] > 0.01


def test_crfboot_intervals_cover_and_have_width():
    y, D, X, tau = _hetero(0, n=800)
    out = causal_forest_bootstrap(y, D, X, B=12, n_trees=40, min_leaf=15, seed=0)
    assert np.all(out["ci_low"] <= out["cate"])
    assert np.all(out["cate"] <= out["ci_high"])
    assert np.mean(out["se"]) > 0
    lo, hi = out["ate_ci"]
    assert lo <= out["ate"] <= hi
    with pytest.raises(ValueError):
        causal_forest_bootstrap(y, D, X, B=1)


def test_qbcfgr_shift_direction():
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.normal(size=(n, 2))
    D = (rng.random(n) < 0.5).astype(float)
    y = 2.0 * D + rng.normal(size=n)  # treatment shifts the distribution up
    out = quantile_balanced_cf(y, D, X, quantile=0.5, n_trees=100, min_leaf=20, seed=0)
    assert np.nanmean(out["shift_effect"]) > 0.2  # measured ~0.6
    assert out["threshold"] == pytest.approx(np.median(y))
    with pytest.raises(ValueError):
        quantile_balanced_cf(y, D, X, quantile=1.5)


def test_htgcrf_isotonic_removes_violations():
    y, D, X, tau = _hetero(0, n=900)
    out = hetero_causal_forest(y, D, X, monotone_feature=0, n_trees=80, min_leaf=20, seed=0)
    assert out["violations_before"] > 0
    assert out["violations_after"] == 0
    order = np.argsort(X[:, 0])
    assert np.all(np.diff(out["cate"][order]) >= -1e-9)  # monotone by construction
    dec = hetero_causal_forest(y, D, X, monotone_feature=0, direction=-1, n_trees=80, seed=0)
    assert np.all(np.diff(dec["cate"][order]) <= 1e-9)
    with pytest.raises(ValueError):
        hetero_causal_forest(y, D, X, monotone_feature=9)


def test_csfgrf_and_front_ends():
    rng = np.random.default_rng(0)
    n = 1200
    X = rng.normal(size=(n, 2))
    D = (rng.random(n) < 0.5).astype(float)
    t_event = rng.exponential(np.exp(0.5 * D))  # treatment lengthens survival
    cens = rng.exponential(4.0, size=n)
    time = np.minimum(t_event, cens)
    event = (t_event <= cens).astype(float)
    out = causal_survival_forest(time, event, D, X, n_trees=80, min_leaf=20, seed=0)
    assert out["ate"] > 0  # positive RMST difference
    assert out["horizon"] > 0
    alt = causal_survival_forest_grf(time, event, D, X, n_trees=80, min_leaf=20, seed=0)
    assert alt["ate"] == pytest.approx(out["ate"])
    blp = causal_survival_blp(time, event, D, X, n_trees=80, min_leaf=20, seed=0)
    assert np.isfinite(blp["beta"])
    assert blp["ate"] == pytest.approx(out["ate"])


def test_drlnr_ate_and_cate():
    hits = 0
    for seed in range(4):
        y, D, X, tau = _hetero(seed)
        out = dr_learner(y, D, X, n_folds=5, seed=seed)
        hits += abs(out["ate"] - 1.0) < 0.25
        r = np.corrcoef(out["cate"], tau)[0, 1]
        assert r > 0.8  # linear second stage on a linear CATE
    assert hits >= 3  # measured 4/4
    with pytest.raises(ValueError):
        dr_learner(y, D, X, n_folds=1)


def test_ipsiMed_effects_sum():
    rng = np.random.default_rng(0)
    n = 3000
    c = rng.normal(size=n)
    x = (rng.random(n) < 0.5).astype(float)
    m = 0.8 * x + 0.4 * c + rng.normal(scale=0.6, size=n)
    y = 0.5 * x + 1.0 * m + 0.3 * c + rng.normal(scale=0.6, size=n)
    out = interventional_psi(y, x, m, c=c)
    assert out["overall"] == pytest.approx(out["ide"] + out["iie"])
    assert out["iie"] == pytest.approx(0.8, abs=0.2)  # 0.8 * 1.0
    assert out["ide"] == pytest.approx(0.5, abs=0.2)
    with pytest.raises(ValueError):
        interventional_psi(y, x, m, n_draws=10)


def test_tmle_core_recovers_ate_under_confounding():
    hits = 0
    for seed in range(8):
        y, A, W = _confounded(seed)
        out = tmle_ate(y, A, W)
        naive = y[A == 1].mean() - y[A == 0].mean()
        assert abs(naive - 2.0) > 0.3  # measured ~3.1
        hits += abs(out["ate"] - 2.0) < 0.25
        assert out["ci"][0] <= out["ate"] <= out["ci"][1]
    assert hits >= 7  # measured 8/8


def test_tmlpoy_double_robustness():
    hits = 0
    for seed in range(6):
        y, A, W = _confounded(seed)
        out = tmle_propensity_only(y, A, W)
        # null outcome model, correct propensity: still consistent
        hits += abs(out["ate"] - 2.0) < 0.3
        assert abs(out["ate_full"] - 2.0) < 0.3
    assert hits >= 5  # measured 6/6


def test_tmltrt_truncation_sweep():
    y, A, W = _confounded(0)
    out = tmle_truncation(y, A, W)
    assert out["ate"].size == out["eps"].size
    assert np.all(np.diff(out["n_truncated"]) >= 0)  # more truncation, more units hit
    assert np.all(np.abs(out["ate"] - 2.0) < 0.5)
    with pytest.raises(ValueError):
        tmle_truncation(y, A, W, eps_grid=[0.6])


def test_tmlsen_gamma_bounds_widen():
    y, A, W = _confounded(0)
    out = tmle_sensitivity_unmeasured(y, A, W, gamma_grid=[1.0, 1.5, 3.0])
    widths = out["upper"] - out["lower"]
    assert widths[0] == pytest.approx(0.0, abs=1e-6)  # Gamma = 1 is a point
    assert widths[1] < widths[2]  # bounds widen with Gamma
    assert out["lower"][0] == pytest.approx(out["ate"], abs=1e-6)
    with pytest.raises(ValueError):
        tmle_sensitivity_unmeasured(y, A, W, gamma_grid=[0.5])


def test_tmlqct_quantile_shift():
    rng = np.random.default_rng(0)
    n = 2000
    W = rng.normal(size=(n, 2))
    A = (rng.random(n) < 1 / (1 + np.exp(-W[:, 0]))).astype(float)
    y = 2.0 * A + W[:, 0] + rng.normal(scale=0.5, size=n)
    out = tmle_quantile(y, A, W, quantile=0.5, n_grid=40)
    assert out["qte"] == pytest.approx(2.0, abs=0.6)  # measured ~1.9
    assert np.all(np.diff(out["f1"]) >= -1e-12)  # monotonised
    with pytest.raises(ValueError):
        tmle_quantile(y, A, W, quantile=0.0)


def test_tmlmed_decomposition_adds_up():
    rng = np.random.default_rng(0)
    n = 3000
    W = rng.normal(size=(n, 2))
    A = (rng.random(n) < 1 / (1 + np.exp(-W[:, 0]))).astype(float)
    M = 0.8 * A + 0.4 * W[:, 0] + rng.normal(scale=0.6, size=n)
    y = 0.5 * A + 1.0 * M + 0.3 * W[:, 0] + rng.normal(scale=0.6, size=n)
    out = tmle_mediation(y, A, M, W)
    assert out["total"] == pytest.approx(out["nde"] + out["nie"])  # exact by construction
    assert out["total"] == pytest.approx(1.3, abs=0.3)  # 0.5 direct + 0.8 indirect
    with pytest.raises(ValueError):
        tmle_mediation(y, np.zeros(n), M, W)  # one arm only


def test_tmlivc_late():
    hits = 0
    for seed in range(6):
        rng = np.random.default_rng(seed)
        n = 4000
        W = rng.normal(size=(n, 2))
        Z = (rng.random(n) < 0.5).astype(float)
        typ = rng.choice(["c", "a", "n"], size=n, p=[0.6, 0.2, 0.2])
        D = np.where(typ == "a", 1.0, np.where(typ == "n", 0.0, Z))
        y = 2.0 * D * (typ == "c") + W[:, 0] + rng.normal(scale=0.5, size=n)
        out = tmle_iv(y, D, Z, W)
        hits += abs(out["late"] - 2.0) < 0.4
        assert 0.4 < out["compliance"] < 0.8
        assert out["ci"][0] <= out["late"] <= out["ci"][1]
    assert hits >= 5  # measured 6/6


def test_tmltvc_and_tmllng():
    hits = 0
    for seed in range(4):
        rng = np.random.default_rng(seed)
        n = 3000
        L1 = rng.normal(size=n)
        A1 = (rng.random(n) < 1 / (1 + np.exp(-L1))).astype(float)
        L2 = 0.5 * L1 + 0.7 * A1 + rng.normal(scale=0.7, size=n)
        A2 = (rng.random(n) < 1 / (1 + np.exp(-1.5 * L2))).astype(float)
        y = 1.0 * A1 + 1.0 * A2 + 1.0 * L2 + rng.normal(scale=0.5, size=n)
        A, L = np.c_[A1, A2], np.c_[L1, L2]
        hi = tmle_time_varying_confound(y, A, L, regime=1.0)
        lo = tmle_time_varying_confound(y, A, L, regime=0.0)
        contrast = hi["estimate"] - lo["estimate"]
        hits += abs(contrast - 2.7) < 0.6  # truth 1 + 1 + 0.7
        assert hi["epsilons"].size == 2
        lng = tmle_longitudinal(y, A, L)
        assert lng["estimate"] == pytest.approx(contrast)
    assert hits >= 3


def test_npstm_rmst_difference():
    rng = np.random.default_rng(0)
    n = 2000
    W = rng.normal(size=(n, 2))
    A = (rng.random(n) < 1 / (1 + np.exp(-W[:, 0]))).astype(float)
    t_event = rng.exponential(np.exp(0.6 * A))
    cens = rng.exponential(4.0, size=n)
    time = np.minimum(t_event, cens)
    event = (t_event <= cens).astype(float)
    out = nonparametric_tmle_survival(time, event, A, W)
    assert out["rmst_difference"] > 0  # treatment lengthens survival
    assert out["rmst1"] > out["rmst0"]
    assert out["ci"][0] <= out["rmst_difference"] <= out["ci"][1]
    assert out["horizon"] > 0
