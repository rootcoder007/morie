"""Synthetic control cluster: caussc, scmaba, ascmcl, gscmcl, causscg,
causscss."""

from morie.fn import _array_core as np
import math
import pytest

from morie.fn.ascmcl import augmented_synthetic_control
from morie.fn.caussc import causal_synthetic_control
from morie.fn.causscg import causal_generalised_sc
from morie.fn.causscss import causal_synthetic_subset
from morie.fn.gscmcl import generalized_synthetic_control
from morie.fn.scmaba import synthetic_control_method


def test_caussc_exact_convex_combination():
    X0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = causal_synthetic_control([0.3, 0.7], X0)
    assert out["weights"] == pytest.approx([0.3, 0.7], abs=1e-4)
    assert out["rmse_pre"] == pytest.approx(0.0, abs=1e-4)
    # simplex constraints
    assert out["weights"].sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(out["weights"] >= -1e-9)


def test_caussc_v_weighting_prioritises_predictor():
    # x1 = (1, 0); donor A matches predictor 1, donor B matches predictor 2.
    X0 = np.array([[1.0, 0.0], [1.0, 0.0]])  # columns: A=(1,1), B=(0,0)
    x1 = np.array([1.0, 0.0])
    out_heavy1 = causal_synthetic_control(x1, X0, V=[100.0, 1.0])
    out_heavy2 = causal_synthetic_control(x1, X0, V=[1.0, 100.0])
    wA_heavy1 = out_heavy1["weights"][0]
    wA_heavy2 = out_heavy2["weights"][0]
    # heavier weight on a predictor pulls towards the donor that matches it
    assert wA_heavy1 > wA_heavy2
    # simplex constraints preserved under both V choices
    for res in (out_heavy1, out_heavy2):
        assert res["weights"].sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(res["weights"] >= -1e-9)
    with pytest.raises(ValueError):
        causal_synthetic_control(x1, X0, V=[-1.0, 1.0])


def test_scmaba_recovers_effect_path():
    rng = np.random.default_rng(0)
    T, J, t0 = 40, 8, 25
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    lam = rng.uniform(0, 1, size=(2, J))
    Y0 = f @ lam + rng.normal(scale=0.1, size=(T, J))
    w_true = np.zeros(J)
    w_true[:3] = (0.5, 0.3, 0.2)
    y1 = Y0 @ w_true + rng.normal(scale=0.1, size=T)
    y1[t0:] += 5.0
    out = synthetic_control_method(y1, Y0, t0)
    # SCM returns simplex weights plus per-period gap and synthetic series
    assert out["weights"].shape == (J,)
    assert out["weights"].sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(out["weights"] >= -1e-9)
    assert out["gap"].shape == (T,)
    assert out["synthetic"].shape == (T,)
    assert out["treat_time"] == t0
    assert math.isfinite(out["att"])
    assert math.isfinite(out["rmse_pre"])
    assert out["rmse_pre"] >= 0


def test_ascmcl_corrects_outside_hull():
    rng = np.random.default_rng(0)
    T, J, t0 = 40, 6, 25
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    lam = rng.uniform(0, 1, size=(2, J))
    Y0 = f @ lam + rng.normal(scale=0.1, size=(T, J))
    y1 = 1.5 * Y0[:, 0] - 0.2 * Y0[:, 1] + rng.normal(scale=0.1, size=T)
    y1[t0:] += 5.0
    out = augmented_synthetic_control(y1, Y0, t0)
    # augmented SCM exposes the augmented ATT, the unaugmented ATT and the
    # extrapolation correction
    assert out["weights"].shape == (J,)
    assert out["weights"].sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(out["weights"] >= -1e-9)
    assert out["treat_time"] == t0
    assert math.isfinite(out["att"])
    assert math.isfinite(out["att_scm"])
    assert math.isfinite(out["correction"])
    assert math.isfinite(out["rmse_pre"])


def test_gscmcl_factor_recovery_outside_hull():
    rng = np.random.default_rng(0)
    T, J, t0, r = 50, 20, 30, 2
    f = np.cumsum(rng.normal(size=(T, r)), axis=0)
    lam0 = rng.normal(size=(r, J))
    Y0 = f @ lam0 + rng.normal(scale=0.2, size=(T, J))
    lam1 = np.array([2.0, -1.0])  # far outside the donor hull
    y1 = f @ lam1 + rng.normal(scale=0.2, size=T)
    y1[t0:] += 3.0
    out = generalized_synthetic_control(y1, Y0, t0, r=r)
    # structural returns
    assert out["gap"].shape == (T,)
    assert out["y0_hat"].shape == (T,)
    assert out["loadings"].shape == (r,)
    assert out["r"] == r
    assert out["treat_time"] == t0
    assert math.isfinite(out["att"])
    with pytest.raises(ValueError):
        generalized_synthetic_control(y1, Y0, t0, r=0)


def test_causscg_matches_gscmcl():
    rng = np.random.default_rng(0)
    Y0 = np.cumsum(rng.normal(size=(30, 6)), axis=0)
    y1 = Y0[:, 0] + rng.normal(scale=0.1, size=30)
    a = causal_generalised_sc(y1, Y0, 20, r=2)
    b = generalized_synthetic_control(y1, Y0, 20, r=2)
    # wrapper is just a front-end to gscmcl
    assert a["att"] == pytest.approx(b["att"])
    assert a["gap"] == pytest.approx(b["gap"])


def test_causscss_selects_true_donors():
    rng = np.random.default_rng(0)
    k, J = 30, 12
    X0 = rng.normal(size=(k, J))
    x1 = 0.6 * X0[:, 3] + 0.4 * X0[:, 7] + rng.normal(scale=0.02, size=k)
    out = causal_synthetic_subset(x1, X0, lam=0.05)
    w = out["weights"]
    # structural returns: simplex weights over the selected support
    assert len(w) == J
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert all(vi >= -1e-9 for vi in w)
    assert out["n_selected"] == len(out["support"])
    # weights should be zero outside the support
    support_set = set(int(s) for s in out["support"])
    for j in range(J):
        if j not in support_set:
            assert abs(w[j]) < 1e-8
    # sparse true signal: at least one donor dropped, but not all
    assert 1 <= out["n_selected"] < J
    assert math.isfinite(out["rmse_pre"])
    assert out["rmse_pre"] >= 0
    with pytest.raises(ValueError):
        causal_synthetic_subset(x1, X0, lam=-0.1)


def test_causscss_lam_monotone_sparsity():
    rng = np.random.default_rng(2)
    X0 = rng.normal(size=(20, 10))
    x1 = X0 @ np.full(10, 0.1)
    n_lo = causal_synthetic_subset(x1, X0, lam=0.001)["n_selected"]
    n_hi = causal_synthetic_subset(x1, X0, lam=0.5)["n_selected"]
    assert n_hi <= n_lo
