"""Survival estimation in the survey/design family."""

import numpy as np
import pytest

from morie.fn.smplts import sample_lifetable
from morie.fn.surdrl import survey_dr_estimator
from morie.fn.survbs import survival_bootstrap_se
from morie.fn.survgen import general_estimating_eq_surv
from morie.fn.survipw import ipcw_estimator
from morie.fn.survlts import life_table_smoothed
from morie.fn.survnls import nonlinear_least_squares_surv
from morie.fn.survvar import variance_cox_estimator


def _surv(n=300, seed=0, cens=0.3):
    rng = np.random.default_rng(seed)
    t = rng.exponential(2.0, n)
    c = rng.exponential(2.0 / max(cens, 1e-6), n)
    obs = np.minimum(t, c)
    ev = (t <= c).astype(float)
    return obs, ev


def test_ipcw_corrects_censoring_bias_and_reports_its_weights():
    t, ev = _surv(400, cens=0.5)
    out = ipcw_estimator(t, ev, tau=float(np.quantile(t, 0.8)))
    assert 0.0 <= out["estimate"] <= 1.5
    # the method's known failure mode is reported, not hidden
    assert out["max_weight"] >= 1.0
    assert out["effective_n"] <= out["n_used"]
    assert out["tau"] == pytest.approx(float(np.quantile(t, 0.8)))
    with pytest.raises(ValueError):
        ipcw_estimator(t, ev * 2)
    with pytest.raises(ValueError):
        ipcw_estimator(-t, ev)


def test_cox_variance_and_the_robust_sandwich_diagnostic():
    rng = np.random.default_rng(1)
    n = 400
    z = rng.standard_normal(n)
    t = rng.exponential(1.0, n) / np.exp(0.8 * z)
    ev = (rng.random(n) > 0.2).astype(float)
    out = variance_cox_estimator(np.array([0.8]), z, t, ev, robust=True)
    assert out["se"][0] > 0
    assert out["robust_se"][0] > 0
    # under a correctly specified model the two agree closely, which
    # is what makes their divergence a diagnostic
    assert 0.5 < out["ratio"] < 2.0
    assert out["n_events"] == int(ev.sum())
    plain = variance_cox_estimator(np.array([0.8]), z, t, ev)
    assert plain["robust_se"] is None
    with pytest.raises(ValueError):
        variance_cox_estimator(np.array([0.8]), z, t, np.zeros(n))


def test_bootstrap_se_is_reported_beside_greenwood():
    t, ev = _surv(150, seed=2)
    g = np.quantile(t, [0.2, 0.4, 0.6])
    out = survival_bootstrap_se(t, ev, t_grid=g, B=60, seed=1)
    assert np.all(out["bootstrap_se"] >= 0)
    assert np.all(out["greenwood_se"] >= 0)
    # in the body of the distribution the two should be comparable
    finite = np.isfinite(out["se_ratio"])
    assert np.all((out["se_ratio"][finite] > 0.3) &
                  (out["se_ratio"][finite] < 3.0))
    assert "subject" in out["resample_level"]
    assert np.all(np.diff(out["survival"]) <= 1e-12)
    with pytest.raises(ValueError):
        survival_bootstrap_se(t, ev, B=5)


def test_smoothed_life_table_gives_up_the_npmle():
    t, ev = _surv(200, seed=3)
    out = life_table_smoothed(t, ev)
    assert out["is_npmle"] is False
    assert out["bandwidth"] > 0
    assert np.all(np.isfinite(out["survival_smooth"]))
    # the smooth curve tracks Kaplan-Meier
    ok = np.isfinite(out["survival_smooth"])
    assert np.corrcoef(out["survival_smooth"][ok],
                       out["survival_km"][ok])[0, 1] > 0.95
    with pytest.raises(ValueError):
        life_table_smoothed(t, ev, bandwidth=-1.0)


def test_actuarial_life_table_uses_the_half_withdrawal_correction():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    entered = np.array([100.0, 70.0, 40.0])
    died = np.array([20.0, 20.0, 10.0])
    withdrawn = np.array([10.0, 10.0, 5.0])
    out = sample_lifetable(edges, entered, died, withdrawn)
    # effective exposure is n - w/2, exactly
    assert np.allclose(out["effective_n"], entered - withdrawn / 2)
    assert out["q"][0] == pytest.approx(20.0 / 95.0)
    assert np.all(np.diff(out["survival"]) <= 0)
    assert out["survival"][0] == pytest.approx(1 - 20.0 / 95.0)
    with pytest.raises(ValueError):
        sample_lifetable(edges, entered, np.array([200.0, 20.0, 10.0]), withdrawn)
    with pytest.raises(ValueError):
        sample_lifetable(np.array([0.0, 2.0, 1.0]), entered, died, withdrawn)


def test_nls_fits_the_curve_but_is_not_for_inference():
    rng = np.random.default_rng(4)
    n = 400
    t = rng.weibull(1.5, n) * 2.0
    ev = np.ones(n)
    out = nonlinear_least_squares_surv(t, ev, model="weibull")
    assert out["valid_for_inference"] is False
    assert out["sse"] >= 0
    # the shape parameter is recovered roughly
    assert 0.8 < out["params"][1] < 2.5
    expo = nonlinear_least_squares_surv(t, ev, model="exponential")
    assert expo["params"].size == 1
    # a Weibull with shape != 1 fits better than an exponential
    assert out["sse"] <= expo["sse"] + 1e-9
    with pytest.raises(ValueError):
        nonlinear_least_squares_surv(t, ev, model="lognormal")


def test_gee_survival_inflates_the_se_only_when_clusters_are_informative():
    rng = np.random.default_rng(5)
    n_clu, m = 40, 8
    frail = rng.standard_normal(n_clu) * 1.2      # strong shared frailty
    z = rng.standard_normal(n_clu * m)
    cl = np.repeat(np.arange(n_clu), m)
    t = rng.exponential(1.0, n_clu * m) / np.exp(0.6 * z + frail[cl])
    ev = (rng.random(n_clu * m) > 0.2).astype(float)
    clustered = general_estimating_eq_surv(t, ev, z, cluster=cl)
    assert abs(clustered["beta"][0] - 0.6) < 0.4
    assert clustered["n_clusters"] == n_clu
    assert clustered["variance_inflation"] is not None
    # with a genuine shared frailty the robust se should not be
    # smaller than the model-based one by much
    assert clustered["variance_inflation"] > 0.7
    assert "independence" in clustered["working_model"]


def test_doubly_robust_needs_only_one_correct_model():
    rng = np.random.default_rng(6)
    n = 1200
    X = rng.standard_normal((n, 2))
    e = 1.0 / (1.0 + np.exp(-(0.6 * X[:, 0])))
    D = (rng.random(n) < e).astype(float)
    y = 1.0 + 2.0 * D + 1.5 * X[:, 0] + rng.standard_normal(n)
    # both models available
    both = survey_dr_estimator(y, D, X)
    assert abs(both["ate"] - 2.0) < 0.3
    # a WRONG propensity with the right outcome model still works
    wrong_ps = survey_dr_estimator(y, D, X, ps=np.full(n, 0.5))
    assert abs(wrong_ps["ate"] - 2.0) < 0.4
    assert both["se"] > 0
    assert "unmeasured confounding" in both["does_not_protect_against"]
    with pytest.raises(ValueError):
        survey_dr_estimator(y, np.zeros(n), X)
