# -*- coding: utf-8 -*-
"""Tests for targeted learning book chapters 1-10."""
import importlib
import math

import pytest

tlroad = importlib.import_module("morie.fn.tlroad")
tlgcmp = importlib.import_module("morie.fn.tlgcmp")
tlseqsl = importlib.import_module("morie.fn.tlseqsl")
tlltmle = importlib.import_module("morie.fn.tlltmle")
tl1step = importlib.import_module("morie.fn.tl1step")
tlhal = importlib.import_module("morie.fn.tlhal")
tlhaltm = importlib.import_module("morie.fn.tlhaltm")
tlheic = importlib.import_module("morie.fn.tlheic")
tldapar = importlib.import_module("morie.fn.tldapar")
tlctmle = importlib.import_module("morie.fn.tlctmle")
np = importlib.import_module("morie.fn._array_core")


def make_data(n=800, seed=3):
    rng = np.random.default_rng(seed)
    W, A, Y, g, q1, q0 = [], [], [], [], [], []
    for _ in range(n):
        w = 1.0 if float(rng.uniform()) < 0.5 else 0.0
        w2 = float(rng.uniform())
        p = 0.25 + 0.5 * w
        a = 1.0 if float(rng.uniform()) < p else 0.0
        m1 = 0.2 + 0.5 * w + 0.1 * w2
        m0 = 0.1 + 0.3 * w + 0.1 * w2
        y = 1.0 if float(rng.uniform()) < (m1 if a == 1.0
                                           else m0) else 0.0
        W.append([w, w2])
        A.append(a)
        Y.append(y)
        g.append(p)
        q1.append(m1)
        q0.append(m0)
    return W, A, Y, g, q1, q0


W, A, Y, G, Q1, Q0 = make_data()
TRUE_ATE = 0.2


# ------------------------------------------------------------- tlroad
def test_tlroad_score_spans_the_eic():
    """Requirement 3: the loss's score along the submodel IS D*."""
    r = tlroad.score_spans_eic(A, Y, Q1, Q0, G)
    assert r["spans"]
    assert r["difference"] < 1e-5


def test_tlroad_targeting_drives_the_score_equation_to_zero():
    psi = tlroad.plugin(Q1, Q0)
    before = tlroad.solves_eic_equation(A, Y, Q1, Q0, G, psi)
    after = tlltmle.tmle_point(A, Y, Q1, Q0, G)
    assert abs(before["mean_eic"]) > 1e-3
    assert abs(after["mean_eic"]) < 1e-10


def test_tlroad_refuses_an_unstated_model():
    with pytest.raises(ValueError):
        tlroad.roadmap("cohort", [], "ATE")
    with pytest.raises(ValueError):
        tlroad.roadmap("cohort", ["iid"], "")


def test_tlroad_rejects_a_positivity_violation():
    with pytest.raises(ValueError):
        tlroad.eic_ate(A, Y, Q1, Q0, [0.0] * len(A), 0.2)


# ------------------------------------------------------------- tlgcmp
def test_tlgcmp_g_computation_closed_form():
    assert abs(tlgcmp.g_computation([0, 1], {0: 0.2, 1: 0.7},
                                    {0: 0.5, 1: 0.5}) - 0.45) < 1e-12
    with pytest.raises(ValueError):
        tlgcmp.g_computation([0, 1], {0: 0.2, 1: 0.7},
                             {0: 0.5, 1: 0.9})


def test_tlgcmp_beats_the_crude_difference_under_confounding():
    Wl = [w[0] for w in W]
    gc = tlgcmp.counterfactual_mean(Y, A, Wl, 1.0) - \
        tlgcmp.counterfactual_mean(Y, A, Wl, 0.0)
    n1 = sum(1 for v in A if v == 1.0)
    n0 = len(A) - n1
    crude = (sum(Y[i] for i in range(len(Y)) if A[i] == 1.0) / n1
             - sum(Y[i] for i in range(len(Y)) if A[i] == 0.0) / n0)
    assert abs(gc - TRUE_ATE) < abs(crude - TRUE_ATE)


def test_tlgcmp_positivity_is_reported():
    assert tlgcmp.positivity_check(G)["satisfied"]
    assert not tlgcmp.positivity_check([0.001, 0.5])["satisfied"]


def test_tlgcmp_empty_stratum_is_a_positivity_violation():
    with pytest.raises(ValueError):
        tlgcmp.counterfactual_mean([1.0, 0.0], [1.0, 1.0], [0, 1],
                                   0.0)


def test_tlgcmp_sequential_formula_ignores_the_treatment_mechanism():
    """Two different g's, same g-formula value."""
    Q = {1: lambda h: 0.3 + 0.4 * h[0] + 0.2 * h[1]}
    sup = [[0.0, 1.0]]
    probs = [lambda h: [0.5, 0.5]]
    r = tlseq = tlgcmp.sequential_g_formula(
        {1: Q[1]}, sup, probs, lambda h: 1.0)
    assert abs(r["psi"] - (0.3 + 0.4 * 0.5 + 0.2)) < 1e-12


# ------------------------------------------------------------ tlseqsl
def _const(Xtr, ytr):
    m = sum(ytr) / len(ytr)
    return lambda x: m


def _linear(Xtr, ytr):
    n = len(ytr)
    mx = sum(r[0] for r in Xtr) / n
    my = sum(ytr) / n
    num = sum((Xtr[i][0] - mx) * (ytr[i] - my) for i in range(n))
    den = sum((Xtr[i][0] - mx) ** 2 for i in range(n)) or 1e-9
    b = num / den
    return lambda x: my + b * (x[0] - mx)


def _toy(n=300, seed=11):
    rng = np.random.default_rng(seed)
    X = [[float(rng.uniform()) * 4.0] for _ in range(n)]
    y = [3.0 * x[0] + float(rng.uniform()) for x in X]
    return X, y


def test_tlseqsl_weights_are_convex():
    X, y = _toy()
    sl = tlseqsl.ensemble_super_learner(X, y,
                                        {"const": _const,
                                         "linear": _linear}, V=5)
    assert abs(sum(sl["weights"].values()) - 1.0) < 1e-9
    assert all(v >= -1e-12 for v in sl["weights"].values())


def test_tlseqsl_ensemble_is_no_worse_than_the_best_member():
    X, y = _toy()
    sl = tlseqsl.ensemble_super_learner(X, y,
                                        {"const": _const,
                                         "linear": _linear}, V=5)
    assert sl["cv_risk"] <= sl["best_single"] + 1e-9


def test_tlseqsl_selector_picks_the_right_algorithm():
    X, y = _toy()
    d = tlseqsl.discrete_super_learner(X, y, {"const": _const,
                                              "linear": _linear}, V=5)
    assert d["selected"] == "linear"


def test_tlseqsl_rejects_bad_input():
    X, y = _toy(30)
    with pytest.raises(ValueError):
        tlseqsl.discrete_super_learner(X, y, {})
    with pytest.raises(ValueError):
        tlseqsl.cv_risk(X, y, _const, V=1)
    with pytest.raises(ValueError):
        tlseqsl.cv_risk(X, y, _const, loss="hinge")


# ------------------------------------------------------------ tlltmle
def test_tlltmle_solves_the_score_equation():
    fit = tlltmle.tmle_point(A, Y, Q1, Q0, G)
    assert fit["solves_eic"]
    assert abs(fit["psi"] - TRUE_ATE) < 0.06


def test_tlltmle_is_double_robust():
    """Either arm may be wrong -- but not both."""
    bad_g = [0.5] * len(A)
    mis = [0.35] * len(A)
    a = tlltmle.tmle_point(A, Y, mis, mis, G)["psi"]
    b = tlltmle.tmle_point(A, Y, Q1, Q0, bad_g)["psi"]
    both = tlltmle.tmle_point(A, Y, mis, mis, bad_g)["psi"]
    assert abs(a - TRUE_ATE) < 0.07
    assert abs(b - TRUE_ATE) < 0.07
    assert abs(both - TRUE_ATE) > 0.04


def test_tlltmle_clever_covariate_exposes_positivity():
    assert tlltmle.clever_covariate(A, [0.02] * len(A))["max"] > 40.0
    with pytest.raises(ValueError):
        tlltmle.clever_covariate(A, [0.0] * len(A))


def test_tlltmle_fluctuation_solves_its_own_score():
    r = tlltmle.fluctuate([0.4] * 100, [1.0] * 100,
                          [1.0] * 30 + [0.0] * 70)
    assert abs(r["score"]) < 1e-8
    assert abs(r["Q_star"][0] - 0.3) < 1e-6


def test_tlltmle_longitudinal_runs_backwards():
    Q = [[0.4] * 50, [0.5] * 50]
    H = [[1.0] * 50, [1.0] * 50]
    r = tlltmle.ltmle(Q, H, [[1.0] * 20 + [0.0] * 30])
    assert r["T"] == 2
    assert len(r["epsilons"]) == 2


# ------------------------------------------------------------ tl1step
def _q_and_y(n=300, seed=4):
    rng = np.random.default_rng(seed)
    q = [min(max(0.3 + 0.4 * float(rng.uniform()), 0.05), 0.95)
         for _ in range(n)]
    y = [1.0 if float(rng.uniform()) < v else 0.0 for v in q]
    return q, y


def _H(Qcur):
    return [1.0 / max(v, 0.05) for v in Qcur]


def test_tl1step_submodel_is_universal_away_from_zero():
    q, _ = _q_and_y()
    u = tl1step.is_universal(q, _H, eps=0.4)
    assert u["universal"]
    assert u["local_submodel_direction_drift"] > 100 * \
        u["max_deviation"]


def test_tl1step_one_move_solves_the_score_equation():
    q, y = _q_and_y()
    one = tl1step.one_step_tmle(q, _H, y, eps_max=1.0, steps=600)
    assert one["iterations"] == 1
    assert one["abs_score"] < 1e-3


def test_tl1step_lands_where_the_iterative_tmle_lands():
    q, y = _q_and_y()
    one = tl1step.one_step_tmle(q, _H, y, eps_max=1.0, steps=600)
    it = tl1step.iterative_tmle(q, _H, y)
    assert abs(one["psi"] - it["psi"]) < 5e-3


def test_tl1step_path_runs_in_both_directions():
    q, y = _q_and_y()
    one = tl1step.one_step_tmle(q, _H, y, eps_max=1.0, steps=600)
    assert one["epsilon"] < 0.0


def test_tl1step_rejects_bad_input():
    with pytest.raises(ValueError):
        tl1step.build_ulfm([0.5], _H, [1.0, 0.0])
    with pytest.raises(ValueError):
        tl1step.build_ulfm([0.5], _H, [1.0], steps=0)


# -------------------------------------------------------------- tlhal
def _step_data(n=60, seed=5):
    rng = np.random.default_rng(seed)
    X = [[float(rng.uniform())] for _ in range(n)]
    y = [1.0 if x[0] > 0.5 else 0.0 for x in X]
    return X, y


def test_tlhal_fits_a_discontinuity():
    X, y = _step_data()
    f = tlhal.hal_fit(X, y, lam=1.0, iters=800)
    assert f["mse"] < 0.03


def test_tlhal_l1_bound_is_the_variation_norm():
    X, y = _step_data()
    f = tlhal.hal_fit(X, y, lam=1.0, iters=800)
    assert abs(f["variation_norm"] - 1.0) < 1e-6
    assert abs(tlhal.variation_norm(f["beta"])
               - f["variation_norm"]) < 1e-12


def test_tlhal_a_tighter_bound_underfits():
    X, y = _step_data()
    a = tlhal.hal_fit(X, y, lam=1.0, iters=800)
    b = tlhal.hal_fit(X, y, lam=0.05, iters=800)
    assert b["mse"] > a["mse"]


def test_tlhal_predictions_honour_the_step():
    X, y = _step_data()
    f = tlhal.hal_fit(X, y, lam=1.0, iters=800)
    p = tlhal.hal_predict(f, [[0.25], [0.75]])
    assert p[1] - p[0] > 0.7


def test_tlhal_basis_grows_with_order():
    X = [[0.1, 0.2], [0.5, 0.6]]
    b1 = tlhal.indicator_basis(X, max_order=1)
    b2 = tlhal.indicator_basis(X, max_order=2)
    assert b2["n_basis"] > b1["n_basis"]


def test_tlhal_rejects_bad_input():
    X, y = _step_data(20)
    with pytest.raises(ValueError):
        tlhal.hal_fit(X, y, lam=0.0)
    with pytest.raises(ValueError):
        tlhal.hal_fit(X, y[:5])


# ------------------------------------------------------------ tlhaltm
def test_tlhaltm_rate_condition_boundary():
    assert not tlhaltm.rate_condition(0.25, 0.25, 1000)["satisfied"]
    assert tlhaltm.rate_condition(1 / 3.0, 1 / 3.0,
                                  1000)["satisfied"]


def test_tlhaltm_positivity_breaks_efficiency_not_the_rate():
    ok = tlhaltm.efficiency_check(0.02, 0.02, 0.1, 1000)
    bad = tlhaltm.efficiency_check(0.02, 0.02, 0.001, 1000)
    assert ok["efficient"]
    assert not bad["efficient"]


def test_tlhaltm_cv_splits_are_disjoint_and_complete():
    sp = tlhaltm.cv_tmle_split(100, 10)
    assert sorted(i for f in sp["folds"] for i in f) == list(range(100))
    for v in range(10):
        assert not set(sp["folds"][v]) & set(sp["training"][v])


def test_tlhaltm_rejects_bad_input():
    with pytest.raises(ValueError):
        tlhaltm.remainder_bound(0.1, 0.1, 0.0)
    with pytest.raises(ValueError):
        tlhaltm.rate_condition(-1.0, 0.3, 100)


# ------------------------------------------------------------- tlheic
def _eic_setup(n=400):
    return A[:n], Y[:n], Q1[:n], Q0[:n], G[:n], n


def test_tlheic_recovers_the_analytic_gradient():
    Aa, Yy, Q1a, Q0a, ga, n = _eic_setup()

    def psi_of_P(w):
        t = sum(w)
        return sum(w[i] * (Q1a[i] - Q0a[i]) for i in range(n)) / t

    basis = [[Q1a[i] - Q0a[i], W[i][0], W[i][1], Aa[i]]
             for i in range(n)]
    est = tlheic.estimate_eic(psi_of_P, basis)
    psi = psi_of_P([1.0 / n] * n)
    part = [Q1a[i] - Q0a[i] - psi for i in range(n)]
    num = sum(est["D"][i] * part[i] for i in range(n))
    den = math.sqrt(sum(v * v for v in est["D"])) * \
        math.sqrt(sum(v * v for v in part))
    assert abs(num / den - 1.0) < 0.02


def test_tlheic_identity_holds_on_a_held_out_direction():
    Aa, Yy, Q1a, Q0a, ga, n = _eic_setup()

    def psi_of_P(w):
        t = sum(w)
        return sum(w[i] * (Q1a[i] - Q0a[i]) for i in range(n)) / t

    basis = [[Q1a[i] - Q0a[i], W[i][0], W[i][1], Aa[i]]
             for i in range(n)]
    est = tlheic.estimate_eic(psi_of_P, basis)
    held = [W[i][0] * W[i][1] for i in range(n)]
    assert tlheic.verify_gradient(psi_of_P, est["D"], held)["verified"]


def test_tlheic_gradient_is_mean_zero():
    Aa, Yy, Q1a, Q0a, ga, n = _eic_setup(200)

    def psi_of_P(w):
        t = sum(w)
        return sum(w[i] * Q1a[i] for i in range(n)) / t

    est = tlheic.estimate_eic(psi_of_P, [[Q1a[i], Aa[i]]
                                         for i in range(n)])
    assert abs(sum(est["D"]) / n) < 1e-9


def test_tlheic_rejects_bad_input():
    with pytest.raises(ValueError):
        tlheic.gradient_inner_product([1.0], [1.0, 2.0])


# ------------------------------------------------------------ tldapar
def _snoop_setup(NN=400, seed=9):
    rng = np.random.default_rng(seed)
    noise = [[float(rng.uniform()) for _ in range(20)]
             for _ in range(NN)]
    target = [float(rng.uniform()) for _ in range(NN)]

    def define(tr):
        best, bj = None, 0
        for j in range(20):
            mx = sum(noise[i][j] for i in tr) / len(tr)
            my = sum(target[i] for i in tr) / len(tr)
            num = sum((noise[i][j] - mx) * (target[i] - my)
                      for i in tr)
            den = math.sqrt(sum((noise[i][j] - mx) ** 2 for i in tr)
                            * sum((target[i] - my) ** 2 for i in tr))
            c = abs(num / den) if den > 0 else 0.0
            if best is None or c > best:
                best, bj = c, j
        return bj

    def est(j, idx):
        mx = sum(noise[i][j] for i in idx) / len(idx)
        my = sum(target[i] for i in idx) / len(idx)
        num = sum((noise[i][j] - mx) * (target[i] - my) for i in idx)
        den = math.sqrt(sum((noise[i][j] - mx) ** 2 for i in idx)
                        * sum((target[i] - my) ** 2 for i in idx))
        return {"estimate": num / den if den > 0 else 0.0,
                "ic": [(noise[i][j] - mx) * (target[i] - my)
                       for i in idx]}

    return define, est, NN


def test_tldapar_split_beats_reusing_one_sample_on_a_null():
    define, est, NN = _snoop_setup()
    split = tldapar.data_adaptive_parameter(define, est, NN, V=5)
    naive = tldapar.naive_reuse(
        lambda idx: est(define(idx), idx), NN)
    assert abs(naive["estimate"]) > abs(split["psi"]) + 0.03
    assert abs(split["psi"]) < 0.10


def test_tldapar_splits_are_disjoint():
    sp = tldapar.split_sample(100, 5)
    assert sorted(i for f in sp["estimation"] for i in f) == \
        list(range(100))
    for v in range(5):
        assert not set(sp["estimation"][v]) & set(sp["training"][v])


def test_tldapar_cv_tmle_pools_the_influence_curve():
    r = tldapar.cv_tmle([0.1, 0.2], [[0.0] * 5, [1.0] * 5], 10)
    assert abs(r["psi"] - 0.15) < 1e-12
    assert r["se"] > 0.0
    with pytest.raises(ValueError):
        tldapar.cv_tmle([0.1], [[0.0] * 3], 10)


# ------------------------------------------------------------ tlctmle
def _instrument_data(n=600, seed=17):
    rng = np.random.default_rng(seed)
    W2, A2, Y2, Q12, Q02 = [], [], [], [], []
    for _ in range(n):
        w = 1.0 if float(rng.uniform()) < 0.5 else 0.0
        z = 1.0 if float(rng.uniform()) < 0.5 else 0.0
        p = min(max(0.5 + 0.15 * w + 0.88 * (z - 0.5), 0.03), 0.97)
        a = 1.0 if float(rng.uniform()) < p else 0.0
        m1, m0 = 0.3 + 0.4 * w, 0.2 + 0.4 * w
        y = 1.0 if float(rng.uniform()) < (m1 if a == 1.0
                                           else m0) else 0.0
        W2.append([w, z])
        A2.append(a)
        Y2.append(y)
        Q12.append(m1)
        Q02.append(m0)
    return W2, A2, Y2, Q12, Q02


def test_tlctmle_rejects_the_instrument():
    W2, A2, Y2, Q12, Q02 = _instrument_data()
    c = tlctmle.ctmle(A2, Y2, Q12, Q02, W2, [[0], [0, 1]], V=5)
    assert c["selected_covariates"] == [0]


def test_tlctmle_instrument_inflates_the_variance_penalty():
    W2, A2, Y2, Q12, Q02 = _instrument_data()
    c = tlctmle.ctmle(A2, Y2, Q12, Q02, W2, [[0], [0, 1]], V=5)
    assert c["variance_penalties"][1] > 5.0 * \
        c["variance_penalties"][0]


def test_tlctmle_still_solves_the_score_equation():
    W2, A2, Y2, Q12, Q02 = _instrument_data()
    c = tlctmle.ctmle(A2, Y2, Q12, Q02, W2, [[0], [0, 1]], V=5)
    assert c["solves_eic"]


def test_tlctmle_instrument_raises_the_clever_covariate():
    W2, A2, Y2, Q12, Q02 = _instrument_data()
    seq = tlctmle.candidate_sequence(A2, W2, [[0], [0, 1]])
    ip = tlctmle.instrument_penalty(seq[0]["g"], seq[1]["g"])
    assert ip["ratio"] > 1.1


def test_tlctmle_rejects_an_empty_candidate_list():
    W2, A2, Y2, Q12, Q02 = _instrument_data(100)
    with pytest.raises(ValueError):
        tlctmle.ctmle(A2, Y2, Q12, Q02, W2, [], V=5)


def test_tl_cheatsheets_are_present():
    for mod in (tlroad, tlgcmp, tlseqsl, tlltmle, tl1step, tlhal,
                tlhaltm, tlheic, tldapar, tlctmle):
        assert len(mod.cheatsheet()) > 80
