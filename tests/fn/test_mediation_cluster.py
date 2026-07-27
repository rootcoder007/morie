"""Mediation cluster: sensIM, rhomed, snsmed, causmedi, medstg, seqM,
tdmed, mcausm, pscme, nemed, immid, weakid, medSEM, mlmMd, longMd,
countMd, survmd, baymed, medML, dmlMed, mssm."""

import numpy as np
import pytest

from morie.fn.baymed import bayes_mediation
from morie.fn.causmedi import causal_mediation_imai
from morie.fn.countMd import count_mediation
from morie.fn.dmlMed import dml_mediation_orthogonal
from morie.fn.immid import index_moderated_mediation
from morie.fn.longMd import longitudinal_mediation
from morie.fn.mcausm import multi_mediator_causal
from morie.fn.medML import ml_mediation_dml
from morie.fn.medSEM import sem_mediation
from morie.fn.medstg import sequential_mediation
from morie.fn.mlmMd import multilevel_mediation
from morie.fn.mssm import marginal_structural_med
from morie.fn.nemed import nested_counterfactual_mediation
from morie.fn.pscme import path_specific_causal_effect
from morie.fn.rhomed import rho_critical_mediation
from morie.fn.sensIM import imai_sensitivity_rho
from morie.fn.seqM import sequential_mediators
from morie.fn.snsmed import sensitivity_mediation
from morie.fn.survmd import survival_mediation
from morie.fn.tdmed import two_dimensional_mediation
from morie.fn.weakid import weak_identification_mediation


def _simple(seed, n=2000, a=0.8, b=1.5, cp=0.7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m = a * x + rng.normal(scale=0.7, size=n)
    y = cp * x + b * m + rng.normal(scale=0.7, size=n)
    return x, m, y


def test_sensIM_theorem2_shape():
    hits = 0
    for seed in range(6):
        x, m, y = _simple(seed)
        out = imai_sensitivity_rho(x, m, y)
        # at rho = 0 the formula must reduce to the product estimate 0.8*1.5
        hits += abs(out["acme_0"] - 1.2) < 0.1
        # the curve is decreasing in rho and crosses zero at rho_tilde
        assert np.all(np.diff(out["acme"]) < 0)
        at_crit = imai_sensitivity_rho(x, m, y, rho_grid=[out["rho_critical"]])
        assert at_crit["acme"][0] == pytest.approx(0.0, abs=1e-9)
    assert hits == 6  # measured 6/6
    with pytest.raises(ValueError):
        imai_sensitivity_rho(*_simple(0), rho_grid=[1.0])


def test_rhomed_and_snsmed():
    x, m, y = _simple(1)
    r = rho_critical_mediation(x, m, y)
    s = imai_sensitivity_rho(x, m, y)
    assert r["rho_critical"] == pytest.approx(s["rho_tilde"])
    assert r["abs_rho_critical"] == pytest.approx(abs(s["rho_tilde"]))
    sn = sensitivity_mediation(x, m, y, rho=[0.0, 0.3])
    assert sn["acme"][0] == pytest.approx(s["acme_0"])
    assert sn["acme"][1] < sn["acme"][0]


def test_causmedi_recovery_and_ci_coverage():
    covered = 0
    for seed in range(12):
        x, m, y = _simple(seed, n=1200)
        out = causal_mediation_imai(x, m, y, n_boot=200, seed=seed)
        assert out["acme"] == pytest.approx(1.2, abs=0.15)
        assert out["ade"] == pytest.approx(0.7, abs=0.15)
        assert out["total"] == pytest.approx(out["acme"] + out["ade"])
        lo, hi = out["acme_ci"]
        covered += lo <= 1.2 <= hi
    # nominal 95% percentile interval; measured coverage 28/30 over a
    # wider seed sweep, so require 10/12 here rather than a point value
    assert covered >= 10


def test_medstg_serial_paths():
    hits = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n = 3000
        x = rng.normal(size=n)
        m1 = 0.6 * x + rng.normal(scale=0.6, size=n)
        m2 = 0.4 * x + 0.5 * m1 + rng.normal(scale=0.6, size=n)
        y = 0.3 * x + 0.7 * m1 + 0.9 * m2 + rng.normal(scale=0.6, size=n)
        out = sequential_mediation(x, m1, m2, y)
        ok = (
            abs(out["direct"] - 0.3) < 0.06
            and abs(out["via_m1"] - 0.6 * 0.7) < 0.06
            and abs(out["via_m2"] - 0.4 * 0.9) < 0.06
            and abs(out["serial"] - 0.6 * 0.5 * 0.9) < 0.06
        )
        hits += ok
        assert out["total"] == pytest.approx(out["direct"] + out["indirect_total"])
        # seqM is the same model with the y-first signature
        sq = sequential_mediators(y, x, m1, m2)
        assert sq["serial"] == pytest.approx(out["serial"])
        # pscme names the same four paths
        ps = path_specific_causal_effect(x, m1, m2, y)
        assert ps["paths"]["X->M1->M2->Y"] == pytest.approx(out["serial"])
        assert ps["total"] == pytest.approx(out["total"])
    assert hits >= 7  # measured 8/8


def test_tdmed_and_mcausm_parallel():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=n)
    m1 = 0.7 * x + rng.normal(scale=0.6, size=n)
    m2 = -0.4 * x + rng.normal(scale=0.6, size=n)
    y = 0.2 * x + 1.0 * m1 + 0.5 * m2 + rng.normal(scale=0.6, size=n)
    td = two_dimensional_mediation(x, m1, m2, y)
    assert td["indirect_m1"] == pytest.approx(0.7, abs=0.06)
    assert td["indirect_m2"] == pytest.approx(-0.2, abs=0.06)
    assert td["contrast"] == pytest.approx(td["indirect_m1"] - td["indirect_m2"])
    mc = multi_mediator_causal(x, np.c_[m1, m2], y)
    assert mc["indirect"] == pytest.approx([td["indirect_m1"], td["indirect_m2"]])
    assert mc["indirect_total"] == pytest.approx(td["indirect_total"])
    assert mc["k"] == 2


def test_nemed_interaction():
    # y has an X*M interaction of 0.5 on top of theta1 = 0.4, theta2 = 1.0
    rng = np.random.default_rng(0)
    n = 6000
    x = (rng.random(n) < 0.5).astype(float)
    m = 0.9 * x + rng.normal(scale=0.7, size=n)
    y = 0.4 * x + 1.0 * m + 0.5 * x * m + rng.normal(scale=0.7, size=n)
    out = nested_counterfactual_mediation(x, m, y)
    # E[M | x = 0] = 0, so NDE = theta1; NIE = (theta2 + theta3) * beta1
    assert out["nde"] == pytest.approx(0.4, abs=0.1)
    assert out["nie"] == pytest.approx(1.5 * 0.9, abs=0.1)
    assert out["te"] == pytest.approx(out["nde"] + out["nie"])
    assert out["interaction"] == pytest.approx(0.5, abs=0.1)


def test_immid_index():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    w = rng.normal(size=n)
    m = 0.3 * x + 0.2 * w + 0.6 * x * w + rng.normal(scale=0.6, size=n)
    y = 0.2 * x + 1.2 * m + rng.normal(scale=0.6, size=n)
    out = index_moderated_mediation(x, m, y, w, w_values=[-1.0, 0.0, 1.0])
    assert out["index"] == pytest.approx(0.6 * 1.2, abs=0.1)
    assert out["conditional_indirect"][1] == pytest.approx(0.3 * 1.2, abs=0.1)
    # the conditional effect is linear in W with slope = index
    d = np.diff(out["conditional_indirect"])
    assert d == pytest.approx([out["index"], out["index"]])


def test_weakid_flags():
    strong = weak_identification_mediation(0.5, 0.5, 0.05, 0.05)
    assert strong["weakly_identified"] is False
    assert strong["sobel_se"] == pytest.approx(np.sqrt(0.5**2 * 0.05**2 * 2))
    weak = weak_identification_mediation(0.5, 0.05, 0.05, 0.05)
    assert weak["weak_b"] is True and weak["weakly_identified"] is True
    with pytest.raises(ValueError):
        weak_identification_mediation(1.0, 1.0, 0.0, 0.1)


def test_medSEM_path_enumeration():
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=n)
    M = 0.8 * X + rng.normal(scale=0.6, size=n)
    Y = 0.5 * X + 1.2 * M + rng.normal(scale=0.6, size=n)
    out = sem_mediation({"M": ["X"], "Y": ["X", "M"]}, {"X": X, "M": M, "Y": Y})
    assert out["paths"]["X->Y"] == pytest.approx(0.5, abs=0.06)
    assert out["paths"]["X->M->Y"] == pytest.approx(0.96, abs=0.08)
    assert out["total_effects"]["X"] == pytest.approx(
        out["paths"]["X->Y"] + out["paths"]["X->M->Y"]
    )
    assert 0 < out["r_squared"]["Y"] < 1
    with pytest.raises(ValueError):
        sem_mediation({"Y": ["Z"]}, {"Y": Y})


def test_mlmMd_within_between_differ():
    rng = np.random.default_rng(0)
    J, npc = 60, 40
    n = J * npc
    cl = np.repeat(np.arange(J), npc)
    # between: a = 1.0, b = 1.0; within: a = 0.2, b = 0.2
    xb = rng.normal(size=J)[cl]
    xw = rng.normal(size=n)
    x = xb + xw
    m = 1.0 * xb + 0.2 * xw + rng.normal(scale=0.3, size=n)
    y = 1.0 * (1.0 * xb) + 0.2 * (0.2 * xw) + 0.0 * x + rng.normal(scale=0.3, size=n)
    m = 1.0 * xb + 0.2 * xw + rng.normal(scale=0.3, size=n)
    y = 1.0 * m * 0 + rng.normal(scale=0.3, size=n)
    # rebuild cleanly: y depends on m with different slopes by level
    mb = np.array([m[cl == j].mean() for j in range(J)])[cl]
    mw = m - mb
    y = 1.0 * mb + 0.2 * mw + rng.normal(scale=0.3, size=n)
    out = multilevel_mediation(y, x, m, cl)
    assert out["indirect_between"] == pytest.approx(1.0 * 1.0, abs=0.2)
    assert out["indirect_within"] == pytest.approx(0.2 * 0.2, abs=0.05)
    assert out["indirect_between"] > out["indirect_within"]
    assert out["n_clusters"] == J
    with pytest.raises(ValueError):
        multilevel_mediation(y[:10], x[:10], m[:10], np.zeros(10))


def test_longMd_cross_lagged():
    rng = np.random.default_rng(0)
    n = 3000
    x1 = rng.normal(size=n)
    m1 = rng.normal(size=n)
    y1 = rng.normal(size=n)
    m2 = 0.5 * x1 + 0.4 * m1 + rng.normal(scale=0.5, size=n)
    y2 = 0.3 * m1 + 0.4 * y1 + rng.normal(scale=0.5, size=n)
    y3 = 0.9 * m2 + 0.1 * x1 + 0.4 * y2 + rng.normal(scale=0.5, size=n)
    out = longitudinal_mediation(np.c_[x1, x1, x1], np.c_[m1, m2, m2], np.c_[y1, y2, y3])
    assert out["a"] == pytest.approx(0.5, abs=0.06)
    assert out["b"] == pytest.approx(0.9, abs=0.06)
    assert out["indirect"] == pytest.approx(0.45, abs=0.08)
    assert out["ar_y"] == pytest.approx(0.4, abs=0.06)
    with pytest.raises(ValueError):
        longitudinal_mediation(np.c_[x1, x1], np.c_[m1, m2], np.c_[y1, y2])  # 2 waves


def test_countMd_rate_ratios():
    rng = np.random.default_rng(0)
    n = 6000
    x = rng.normal(size=n)
    m = 0.5 * x + rng.normal(scale=0.5, size=n)
    lam = np.exp(0.3 * x + 0.4 * m)
    y = rng.poisson(lam)
    out = count_mediation(y, x, m)
    assert out["coefficients"]["theta1"] == pytest.approx(0.3, abs=0.06)
    assert out["log_nie"] == pytest.approx(0.4 * 0.5, abs=0.06)
    assert out["rr_total"] == pytest.approx(out["rr_nde"] * out["rr_nie"])
    with pytest.raises(ValueError):
        count_mediation([-1.0] * 20, np.zeros(20), np.zeros(20))


def test_survmd_hazard_ratios():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    m = 0.5 * x + rng.normal(scale=0.5, size=n)
    lin = 0.4 * x + 0.6 * m
    t_event = rng.exponential(np.exp(-lin))
    cens = rng.exponential(2.0, size=n)
    time = np.minimum(t_event, cens)
    event = (t_event <= cens).astype(float)
    out = survival_mediation(time, event, x, m)
    assert out["coefficients"]["theta1"] == pytest.approx(0.4, abs=0.1)
    assert out["coefficients"]["theta2"] == pytest.approx(0.6, abs=0.1)
    assert out["hr_total"] == pytest.approx(out["hr_nde"] * out["hr_nie"])
    assert out["n_events"] > 100
    with pytest.raises(ValueError):
        survival_mediation(time, np.zeros(n), x, m)  # no events


def test_baymed_posterior():
    x, m, y = _simple(0, n=1500)
    out = bayes_mediation(x, m, y, n_draws=2000, seed=0)
    assert out["indirect_mean"] == pytest.approx(1.2, abs=0.15)
    lo, hi = out["indirect_ci"]
    assert lo < 1.2 < hi
    assert out["p_direction"] > 0.99
    assert out["draws"].size == 2000
    with pytest.raises(ValueError):
        bayes_mediation(x, m, y, prior_sd=0.0)


def test_medML_crossfit_removes_covariate_confounding():
    hits = 0
    for seed in range(6):
        rng = np.random.default_rng(seed)
        n = 2000
        C = rng.normal(size=(n, 5))
        x = C @ np.array([0.5, -0.3, 0.2, 0.0, 0.1]) + rng.normal(size=n)
        m = 0.8 * x + C @ np.full(5, 0.3) + rng.normal(scale=0.6, size=n)
        y = 0.7 * x + 1.5 * m + C @ np.full(5, -0.2) + rng.normal(scale=0.6, size=n)
        out = ml_mediation_dml(x, m, y, C, n_folds=5, seed=seed)
        hits += abs(out["indirect"] - 1.2) < 0.15 and abs(out["direct"] - 0.7) < 0.15
        assert out["total"] == pytest.approx(out["direct"] + out["indirect"])
        alt = dml_mediation_orthogonal(x, m, y, C, n_folds=5, seed=seed)
        assert alt["indirect"] == pytest.approx(out["indirect"])
    assert hits >= 5  # measured 6/6


def test_mssm_weighted_msm():
    hits = 0
    for seed in range(6):
        rng = np.random.default_rng(seed)
        n = 3000
        c = rng.normal(size=n)
        x = (rng.random(n) < 1 / (1 + np.exp(-c))).astype(float)
        m = 0.8 * x + 0.5 * c + rng.normal(scale=0.6, size=n)
        y = 0.4 * x + 1.0 * m + 0.5 * c + rng.normal(scale=0.6, size=n)
        out = marginal_structural_med(x, m, y, c=c)
        hits += abs(out["nie"] - 0.8) < 0.25
        assert out["te"] == pytest.approx(out["nde"] + out["nie"])
        assert out["ess"] > 0
    assert hits >= 5  # measured 6/6
    with pytest.raises(ValueError):
        marginal_structural_med(np.zeros(20), np.zeros(20), np.zeros(20))
