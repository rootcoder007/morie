# -*- coding: utf-8 -*-
"""Tests: crfflt, bigtm, bprMF, caltbR, cdaeRC."""
import importlib
import math

import pytest

crfflt = importlib.import_module("morie.fn.crfflt")
bigtm = importlib.import_module("morie.fn.bigtm")
bprMF = importlib.import_module("morie.fn.bprMF")
caltbR = importlib.import_module("morie.fn.caltbR")
cdaeRC = importlib.import_module("morie.fn.cdaeRC")
np = importlib.import_module("morie.fn._array_core")


# ------------------------------------------------------------- crfflt
def test_crfflt_ideal_weights_closed_form():
    w = crfflt.ideal_weights(6.0, 32.0, 5)
    a, b = w["a"], w["b"]
    assert abs(w["B"][0] - (b - a) / math.pi) < 1e-15
    for j in range(1, 6):
        want = (math.sin(j * b) - math.sin(j * a)) / (math.pi * j)
        assert abs(w["B"][j] - want) < 1e-15


def test_crfflt_weights_still_large_at_lag_120():
    """Fig. 1a: truncation bites because the tails do not vanish."""
    w = crfflt.ideal_weights(18.0, 96.0, 130)
    assert abs(w["B"][120]) > 1e-3


def test_crfflt_annihilates_a_constant():
    r = crfflt.cf_filter([7.5] * 60, 6, 32)
    assert r["max_abs_weight_sum"] < 1e-12
    assert max(abs(v) for v in r["cycle"]) < 1e-10


def test_crfflt_symmetric_is_drift_invariant():
    base = [math.sin(2 * math.pi * t / 20.0) for t in range(80)]
    trend = [base[t] + 0.5 * t for t in range(80)]
    s0 = crfflt.cf_filter(base, 6, 32, method="symmetric",
                          p=12)["cycle"]
    s1 = crfflt.cf_filter(trend, 6, 32, method="symmetric",
                          p=12)["cycle"]
    assert max(abs(s0[t] - s1[t]) for t in range(12, 68)) < 1e-9
    a0 = crfflt.cf_filter(base, 6, 32, drift=False)["cycle"]
    a1 = crfflt.cf_filter(trend, 6, 32, drift=False)["cycle"]
    assert max(abs(a0[t] - a1[t]) for t in range(80)) > 1e-3


def test_crfflt_passes_the_band_and_stops_the_rest():
    inb = [math.sin(2 * math.pi * t / 20.0) for t in range(200)]
    outb = [math.sin(2 * math.pi * t / 100.0) for t in range(200)]
    yi = crfflt.cf_filter(inb, 6, 32)["cycle"]
    yo = crfflt.cf_filter(outb, 6, 32)["cycle"]
    assert max(abs(v) for v in yi[50:150]) > 0.9
    assert max(abs(v) for v in yo[50:150]) < 0.2


def test_crfflt_one_sided_ignores_the_future():
    x1 = [math.sin(2 * math.pi * t / 20.0) for t in range(200)]
    x2 = list(x1)
    for t in range(150, 200):
        x2[t] += 5.0
    o1 = crfflt.cf_filter(x1, 6, 32, method="one_sided", drift=False)
    o2 = crfflt.cf_filter(x2, 6, 32, method="one_sided", drift=False)
    assert max(abs(o1["cycle"][t] - o2["cycle"][t])
               for t in range(2, 150)) < 1e-12


def test_crfflt_rejects_bad_input():
    with pytest.raises(ValueError):
        crfflt.ideal_weights(32.0, 6.0, 5)
    with pytest.raises(ValueError):
        crfflt.cf_filter([1.0, 2.0], 6, 32)
    with pytest.raises(ValueError):
        crfflt.cf_filter([1.0] * 20, 6, 32, method="hodrick")
    with pytest.raises(ValueError):
        crfflt.cf_filter([1.0] * 20, 6, 32, method="symmetric", p=15)


# -------------------------------------------------------------- bigtm
def test_bigtm_predictive_equals_the_interpolation():
    d = bigtm.dirichlet_predictive([3.0, 1.0, 0.0], 4.0, 2.0,
                                   [0.5, 0.3, 0.2])
    for i in range(3):
        assert abs(d["predictive"][i] - d["interpolated"][i]) < 1e-15


def test_bigtm_beta_limits():
    lo = bigtm.dirichlet_predictive([3.0, 1.0, 0.0], 4.0, 1e-9,
                                    [0.5, 0.3, 0.2])["predictive"]
    hi = bigtm.dirichlet_predictive([3.0, 1.0, 0.0], 4.0, 1e9,
                                    [0.5, 0.3, 0.2])["predictive"]
    assert abs(lo[0] - 0.75) < 1e-6
    assert abs(hi[0] - 0.5) < 1e-6


def test_bigtm_printed_eq15_disagrees_with_eq13():
    """Recorded, not silently followed."""
    lp = bigtm.lda_predictive([6.0, 2.0], 8.0, 3.0, [0.5, 0.5])
    assert abs(lp["predictive"][0] - lp["eq15_as_printed"][0]) > 1e-3
    assert abs(lp["predictive"][0] - (6.0 + 1.5) / 11.0) < 1e-12


def test_bigtm_context_changes_the_prediction():
    a = bigtm.bigram_topic_predictive([9.0, 1.0], 10.0, 1.0,
                                      [0.5, 0.5])["predictive"]
    b = bigtm.bigram_topic_predictive([1.0, 9.0], 10.0, 1.0,
                                      [0.5, 0.5])["predictive"]
    assert abs(a[0] - b[0]) > 0.5


def test_bigtm_gibbs_separates_planted_topics():
    """The topic must carry what the previous word does not."""
    A = [[0, 1] * 10] * 8
    B = [[0, 2] * 10] * 8
    g = bigtm.gibbs_bigram_topic(A + B, 2, 3, alpha=0.5, beta=0.1,
                                 iters=2000, burn=500, seed=7)
    th = g["theta"]
    ka = th[0].index(max(th[0]))
    kb = th[8].index(max(th[8]))
    assert ka != kb
    assert all(th[d].index(max(th[d])) == ka for d in range(8))
    assert all(th[d].index(max(th[d])) == kb for d in range(8, 16))
    assert min(max(t) for t in th) > 0.9


def test_bigtm_evidence_prefers_the_truth():
    A = [[0, 1] * 10] * 8
    B = [[0, 2] * 10] * 8
    zt = [[0] * 20 for _ in range(8)] + [[1] * 20 for _ in range(8)]
    zs = [[i % 2 for i in range(20)] for _ in range(16)]
    assert bigtm.log_evidence(A + B, 2, 3, zt, beta=0.1) > \
        bigtm.log_evidence(A + B, 2, 3, zs, beta=0.1) + 1.0


def test_bigtm_rejects_bad_input():
    with pytest.raises(ValueError):
        bigtm.dirichlet_predictive([1.0], 1.0, 1.0, [0.5, 0.5])
    with pytest.raises(ValueError):
        bigtm.dirichlet_predictive([1.0, 1.0], 2.0, 1.0, [0.5, 0.9])
    with pytest.raises(ValueError):
        bigtm.dirichlet_predictive([1.0, 1.0], 2.0, -1.0, [0.5, 0.5])
    with pytest.raises(ValueError):
        bigtm.gibbs_bigram_topic([[0, 5]], 2, 3)
    with pytest.raises(ValueError):
        bigtm.bigram_topic_predictive([1.0], 1.0, 1.0, [1.0], prior=3)


# -------------------------------------------------------------- bprMF
POS = {0: [0, 1], 1: [0, 1], 2: [2, 3], 3: [2, 3]}


def test_bprMF_auc_extremes():
    pos = {0: [0, 1]}
    W = [[1.0]]
    good = [[2.0], [2.0], [-1.0], [-1.0]]
    bad = [[-1.0], [-1.0], [2.0], [2.0]]
    tie = [[0.0]] * 4
    assert abs(bprMF.auc(W, good, pos, 4)["auc"] - 1.0) < 1e-12
    assert abs(bprMF.auc(W, bad, pos, 4)["auc"]) < 1e-12
    assert abs(bprMF.auc(W, tie, pos, 4)["auc"]) < 1e-12


def test_bprMF_triple_count():
    r = bprMF.bpr_opt([[1.0], [1.0]], [[0.5]] * 4,
                      {0: [0, 1], 1: [2]}, 4)
    assert r["n_triples"] == 2 * 2 + 1 * 3


def test_bprMF_sigmoid_and_decomposition():
    assert abs(bprMF.sigmoid(0.0) - 0.5) < 1e-15
    assert bprMF.sigmoid(-800.0) >= 0.0
    W = [[0.3, -0.2]]
    H = [[0.5, 0.1], [-0.4, 0.7]]
    x = bprMF.predict(W, H, 0, 0) - bprMF.predict(W, H, 0, 1)
    assert abs(x - (0.3 * 0.9 + (-0.2) * (-0.6))) < 1e-12


def test_bprMF_learns_the_planted_blocks():
    fit = bprMF.learn_bpr(POS, 4, 4, k_dim=4, alpha=0.1, lam=0.005,
                          iters=6000, seed=3)
    assert fit["auc"] > 0.95


def test_bprMF_printed_regularizer_sign_diverges():
    bad = bprMF.learn_bpr(POS, 4, 4, k_dim=4, alpha=0.1, lam=0.5,
                          iters=6000, seed=3,
                          regularizer_sign="paper")
    good = bprMF.learn_bpr(POS, 4, 4, k_dim=4, alpha=0.1, lam=0.5,
                           iters=6000, seed=3)
    assert bad["param_norm"] > 100.0 * good["param_norm"]


def test_bprMF_rejects_bad_input():
    with pytest.raises(ValueError):
        bprMF.learn_bpr({}, 2, 4)
    with pytest.raises(ValueError):
        bprMF.learn_bpr(POS, 4, 1)
    with pytest.raises(ValueError):
        bprMF.learn_bpr(POS, 4, 4, regularizer_sign="either")
    with pytest.raises(ValueError):
        bprMF.auc([[1.0]], [[1.0]] * 2, {0: [0, 1]}, 2)


# ------------------------------------------------------------- caltbR
TABLE1 = [((0.6, 0.4), (0.5, 0.5), 0.0197),
          ((0.6, 0.4), (0.6, 0.4), 0.0),
          ((0.6, 0.4), (0.7, 0.3), 0.0221),
          ((0.7, 0.3), (0.6, 0.4), 0.0212),
          ((0.7, 0.3), (0.7, 0.3), 0.0),
          ((0.7, 0.3), (0.8, 0.2), 0.0275),
          ((0.7, 0.3), (0.69, 0.31), 2.31e-4),
          ((0.7, 0.3), (0.71, 0.29), 2.36e-4)]


@pytest.mark.parametrize("p,q,want", TABLE1)
def test_caltbR_reproduces_steck_table_1(p, q, want):
    got = caltbR.calibration_kl(list(p), list(q))
    assert abs(got - want) <= max(2e-4, 0.02 * want)


def test_caltbR_kl_properties():
    """The three properties Sec. 3 asks a calibration metric to have."""
    assert caltbR.calibration_kl([0.6, 0.4], [0.6, 0.4]) < 1e-12
    assert caltbR.calibration_kl([0.02, 0.98], [0.01, 0.99]) > \
        caltbR.calibration_kl([0.50, 0.50], [0.49, 0.51])
    assert caltbR.calibration_kl([0.3, 0.7], [0.31, 0.69]) < \
        caltbR.calibration_kl([0.3, 0.7], [0.29, 0.71])


def test_caltbR_hellinger_is_defined_at_zeros():
    assert caltbR.calibration_hellinger([0.5, 0.5], [1.0, 0.0]) > 0.0
    assert abs(caltbR.calibration_hellinger([0.5, 0.5],
                                            [0.5, 0.5])) < 1e-12


def test_caltbR_rerank_restores_the_minority_genre():
    PG = [[1.0, 0.0]] * 20 + [[0.0, 1.0]] * 20
    scores = [1.0 - 0.01 * i for i in range(20)] + \
             [0.80 - 0.01 * i for i in range(20)]
    cal = caltbR.calibrated_rerank(scores, PG, [0.7, 0.3], N=10,
                                   lam=0.9)
    assert cal["q"][1] >= 0.25
    assert cal["calibration"] < cal["calibration_uncalibrated"]
    assert cal["score"] < cal["score_uncalibrated"]


def test_caltbR_lambda_zero_is_the_accuracy_ranking():
    PG = [[1.0, 0.0]] * 20 + [[0.0, 1.0]] * 20
    scores = [1.0 - 0.01 * i for i in range(20)] + \
             [0.80 - 0.01 * i for i in range(20)]
    r = caltbR.calibrated_rerank(scores, PG, [0.7, 0.3], N=5, lam=0.0)
    assert r["ranking"] == [0, 1, 2, 3, 4]


def test_caltbR_diversity_prior_admits_an_unplayed_genre():
    dv = caltbR.diversity_prior([0.7, 0.3, 0.0], [1 / 3.0] * 3, 0.3)
    assert dv[2] > 0.09
    assert abs(sum(dv) - 1.0) < 1e-12


def test_caltbR_rejects_bad_input():
    with pytest.raises(ValueError):
        caltbR.calibration_kl([0.5, 0.5], [0.3, 0.3, 0.4])
    with pytest.raises(ValueError):
        caltbR.calibration_kl([0.5, 0.5], [0.5, 0.5], alpha=1.0)
    with pytest.raises(ValueError):
        caltbR.diversity_prior([0.5, 0.5], [0.5, 0.5], 1.5)
    with pytest.raises(ValueError):
        caltbR.calibrated_rerank([1.0], [[1.0]], [1.0], metric="chi2")


# ------------------------------------------------------------- cdaeRC
def test_cdaeRC_corruption_is_unbiased():
    rng = np.random.default_rng(11)
    tot, R = 0.0, 2000
    for _ in range(R):
        tot += sum(cdaeRC.corrupt([1.0] * 20, 0.4, rng))
    assert abs(tot / (R * 20) - 1.0) < 0.03


def test_cdaeRC_losses_match_closed_forms():
    assert abs(cdaeRC.loss(1.0, 0.5, "square") - 0.125) < 1e-15
    assert abs(cdaeRC.loss(1.0, 0.5, "log")
               - math.log(1 + math.exp(-0.5))) < 1e-15
    assert abs(cdaeRC.loss(1.0, 0.5, "hinge") - 0.5) < 1e-15
    assert abs(cdaeRC.loss(1.0, 0.0, "cross_entropy")
               - math.log(2.0)) < 1e-15


def test_cdaeRC_negative_label_must_be_minus_one():
    with pytest.raises(ValueError):
        cdaeRC.loss(0.0, 0.5, "hinge")
    with pytest.raises(ValueError):
        cdaeRC.loss(0.0, 0.5, "log")
    assert cdaeRC.loss(0.0, 0.5, "square") > 0.0


def test_cdaeRC_user_node_separates_identical_inputs():
    W = [[0.5, -0.3] for _ in range(4)]
    z1 = cdaeRC.encode([1.0, 0.0, 1.0, 0.0], W, [0.0, 0.0], [0.0, 0.0])
    z2 = cdaeRC.encode([1.0, 0.0, 1.0, 0.0], W, [2.0, -2.0],
                       [0.0, 0.0])
    assert max(abs(z1[f] - z2[f]) for f in range(2)) > 0.1


def test_cdaeRC_training_reduces_error_and_ranks_the_block():
    pos = {0: [0, 1], 1: [0, 1], 2: [2, 3], 3: [2, 3]}
    m = cdaeRC.fit_cdae(pos, 4, 4, k_dim=4, q=0.2, alpha=0.3,
                        lam=0.001, iters=200, n_neg=2, seed=5)
    assert m["loss_history"][-1] < 0.5 * m["loss_history"][0]
    r = cdaeRC.recommend(m, {0: [0]}, 0, 4, top_k=3)
    assert r["ranking"][0][0] == 1


def test_cdaeRC_rejects_bad_input():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        cdaeRC.corrupt([1.0], 1.0, rng)
    with pytest.raises(ValueError):
        cdaeRC.loss(1.0, 0.5, "huber")
    with pytest.raises(ValueError):
        cdaeRC.fit_cdae({0: [0]}, 1, 1)


def test_rec_cheatsheets_are_present():
    for mod in (crfflt, bigtm, bprMF, caltbR, cdaeRC):
        assert len(mod.cheatsheet()) > 80
