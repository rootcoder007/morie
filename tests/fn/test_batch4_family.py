# -*- coding: utf-8 -*-
"""Tests for the 24-module batch: recommenders, graphs, classics."""
import importlib
import math

import pytest

impFB = importlib.import_module("morie.fn.impFB")
fmFM = importlib.import_module("morie.fn.fmFM")
ffmFM = importlib.import_module("morie.fn.ffmFM")
ncfRS = importlib.import_module("morie.fn.ncfRS")
ngcf = importlib.import_module("morie.fn.ngcf")
gru4r = importlib.import_module("morie.fn.gru4r")
narm = importlib.import_module("morie.fn.narm")
fairRC = importlib.import_module("morie.fn.fairRC")
node2v = importlib.import_module("morie.fn.node2v")
gsage = importlib.import_module("morie.fn.gsageemd")
mpfn = importlib.import_module("morie.fn.mpfn")
egnnL = importlib.import_module("morie.fn.egnnL")
egcn = importlib.import_module("morie.fn.egcn")
gnnEx = importlib.import_module("morie.fn.gnnEx")
grace = importlib.import_module("morie.fn.grace")
gtrf = importlib.import_module("morie.fn.gtrf")
meglt = importlib.import_module("morie.fn.meglt")
polyak = importlib.import_module("morie.fn.polyak")
lsa = importlib.import_module("morie.fn.lsa")
peg = importlib.import_module("morie.fn.prsPEG")
dqnv = importlib.import_module("morie.fn.dqnv")
resnxt = importlib.import_module("morie.fn.resnxt")
mienco = importlib.import_module("morie.fn.mienco")
pratt = importlib.import_module("morie.fn.pratt")
np = importlib.import_module("morie.fn._array_core")

ADJ = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 4], 3: [1, 4], 4: [2, 3]}
R = [[0.0, 3.0, 0.0, 1.0], [2.0, 0.0, 0.0, 4.0],
     [0.0, 0.0, 5.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
POS = {0: [0, 1], 1: [0, 1], 2: [2, 3], 3: [2, 3]}


# -------------------------------------------------------------- impFB
def test_impFB_confidence_and_preference():
    assert impFB.confidence(R)[0][1] == 1.0 + 40.0 * 3.0
    assert impFB.preference(R)[0][1] == 1.0
    assert impFB.preference(R)[0][0] == 0.0


def test_impFB_fast_decomposition_equals_the_naive_form():
    Y = [[0.3, -0.1], [0.5, 0.2], [-0.2, 0.4], [0.1, 0.6]]
    C0, P0 = impFB.confidence(R)[0], impFB.preference(R)[0]
    a = impFB.als_step(Y, C0, P0, 0.1, fast=True)
    b = impFB.als_step(Y, C0, P0, 0.1, fast=False)
    assert max(abs(a[f] - b[f]) for f in range(2)) < 1e-10


def test_impFB_als_reduces_the_cost():
    fit = impFB.fit_wrmf(R, f=2, iters=12, lam=0.1)
    h = fit["cost_history"]
    assert all(h[i] <= h[i - 1] + 1e-6 for i in range(1, len(h)))
    assert h[-1] < h[0]


def test_impFB_explanation_sums_to_the_prediction():
    Y = [[0.3, -0.1], [0.5, 0.2], [-0.2, 0.4], [0.1, 0.6]]
    C0, P0 = impFB.confidence(R)[0], impFB.preference(R)[0]
    ex = impFB.explain(Y, C0, P0, 1, 0.1)
    assert abs(sum(ex["contributions"].values())
               - ex["prediction"]) < 1e-12


def test_impFB_rejects_bad_input():
    with pytest.raises(ValueError):
        impFB.fit_wrmf([[-1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        impFB.confidence(R, alpha=-1.0)


# --------------------------------------------------------------- fmFM
def test_fmFM_linear_time_form_matches_the_double_sum():
    x = [1.0, 0.0, 2.0, 1.0, 0.5]
    w = [0.1, -0.2, 0.3, 0.0, 0.4]
    V = [[0.2, -0.1], [0.3, 0.5], [-0.4, 0.2], [0.1, 0.1],
         [0.5, -0.3]]
    assert abs(fmFM.predict(x, 0.3, w, V)
               - fmFM.predict_naive(x, 0.3, w, V)) < 1e-12


def test_fmFM_gradient_matches_finite_differences():
    x = [1.0, 1.0, 0.0]
    V = [[0.4, 0.1], [-0.2, 0.3], [0.7, 0.5]]
    h = 1e-6
    up = [list(r) for r in V]
    up[0][0] += h
    dn = [list(r) for r in V]
    dn[0][0] -= h
    fd = (fmFM.predict(x, 0.0, [0.0] * 3, up)
          - fmFM.predict(x, 0.0, [0.0] * 3, dn)) / (2 * h)
    assert abs(fd - fmFM.gradient(x, V, 0, 0)) < 1e-6


def test_fmFM_mf_encoding_recovers_matrix_factorisation():
    d = fmFM.design_mf(1, 2, 3, 4)
    V = [[1.0], [2.0], [3.0], [0.5], [1.5], [2.5], [3.5]]
    assert abs(fmFM.predict(d, 0.0, [0.0] * 7, V) - 2.0 * 2.5) < 1e-12


def test_fmFM_rejects_bad_input():
    with pytest.raises(ValueError):
        fmFM.fit_fm([[1.0]], [1.0, 2.0])
    with pytest.raises(ValueError):
        fmFM.fit_fm([[1.0]], [1.0], k_dim=0)


# -------------------------------------------------------------- ffmFM
def test_ffmFM_uses_the_crossed_field_index():
    W = [[[1.0], [2.0]], [[3.0], [4.0]]]
    got = ffmFM.phi([(0, 1.0), (1, 1.0)], [0, 1], W)
    assert abs(got - W[0][1][0] * W[1][0][0]) < 1e-12
    assert abs(got - W[0][0][0] * W[1][1][0]) > 1e-9


def test_ffmFM_parameter_counts():
    assert ffmFM.n_parameters(100, 5, 4) == 2000
    assert ffmFM.n_parameters(100, 5, 4, "fm") == 400


def test_ffmFM_label_must_be_pm_one():
    with pytest.raises(ValueError):
        ffmFM.logistic_loss(0.0, 1.0)
    assert ffmFM.logistic_loss(1.0, 0.0) == math.log(2.0)


# -------------------------------------------------------------- ncfRS
def test_ncfRS_gmf_recovers_matrix_factorisation():
    p, q = [0.5, -0.2, 0.3], [1.0, 2.0, -1.0]
    assert abs(ncfRS.gmf(p, q, None, "identity")
               - sum(p[i] * q[i] for i in range(3))) < 1e-12


def test_ncfRS_learned_h_reweights_dimensions():
    p, q = [0.5, -0.2, 0.3], [1.0, 2.0, -1.0]
    assert abs(ncfRS.gmf(p, q, [2.0, 0.0, 1.0], "identity")
               - (1.0 - 0.3)) < 1e-12


def test_ncfRS_gmf_learns_the_planted_blocks():
    g = ncfRS.fit_gmf(POS, 4, 4, k_dim=4, alpha=0.2, iters=1500,
                      seed=2)
    assert ncfRS.gmf(g["P"][0], g["Q"][1], g["h"]) > \
        ncfRS.gmf(g["P"][0], g["Q"][2], g["h"]) + 0.2


def test_ncfRS_rejects_bad_input():
    with pytest.raises(ValueError):
        ncfRS.gmf([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        ncfRS.gmf([1.0], [1.0], activation="tanh")
    with pytest.raises(ValueError):
        ncfRS.fit_gmf({}, 2, 4)


# --------------------------------------------------------------- ngcf
def test_ngcf_laplacian_coefficient():
    assert abs(ngcf.laplacian_coefficient(4, 9) - 1 / 6.0) < 1e-15
    with pytest.raises(ValueError):
        ngcf.laplacian_coefficient(0, 3)


def test_ngcf_affinity_term_changes_the_message():
    I = [[1.0, 0.0], [0.0, 1.0]]
    a = ngcf.message([2.0, 3.0], [1.0, 4.0], I, I, 1.0, True)
    b = ngcf.message([2.0, 3.0], [1.0, 4.0], I, I, 1.0, False)
    assert abs(a[0] - b[0]) > 1e-9
    assert abs(b[0] - 2.0) < 1e-12
    assert abs(a[0] - 4.0) < 1e-12


def test_ngcf_concatenates_every_order():
    I = [[1.0, 0.0], [0.0, 1.0]]
    E0 = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5],
          [0.2, 0.8]]
    st = ngcf.stack_layers(E0, ADJ, [(I, I), (I, I)])
    assert len(st["final"][0]) == 6


# -------------------------------------------------------------- gru4r
def test_gru4r_session_parallel_batches_reset_slots():
    b = gru4r.session_parallel_batches([[1, 2, 3], [4, 5],
                                        [6, 7, 8, 9]], 2)
    assert b["n_steps"] >= 3
    assert any(any(s["reset"]) for s in b["steps"])


def test_gru4r_top1_regularizer_punishes_inflated_scores():
    lo = gru4r.top1_loss(2.0, [0.0, -0.5])
    hi = gru4r.top1_loss(12.0, [10.0, 9.5])
    assert hi > lo + 0.2
    assert abs(gru4r.top1_loss(2.0, [0.0, -0.5], regularize=False)
               - gru4r.top1_loss(12.0, [10.0, 9.5],
                                 regularize=False)) < 1e-12


def test_gru4r_ranking_metrics():
    assert gru4r.recall_at_k([5, 3, 1], 3, 2) == 1.0
    assert gru4r.recall_at_k([5, 3, 1], 1, 2) == 0.0
    assert abs(gru4r.mrr_at_k([5, 3, 1], 3, 3) - 0.5) < 1e-15


def test_gru4r_rejects_bad_input():
    with pytest.raises(ValueError):
        gru4r.session_parallel_batches([[1]], 1)
    with pytest.raises(ValueError):
        gru4r.bpr_loss(1.0, [])


# --------------------------------------------------------------- narm
def test_narm_bilinear_decoder_parameter_count():
    dp = narm.decoder_parameters(50000, 100, 100)
    assert dp["fully_connected"] == 5000000
    assert dp["bilinear"] == 10000


def test_narm_attention_is_a_distribution():
    H = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    I = [[1.0, 0.0], [0.0, 1.0]]
    a = narm.attention_weights([1.0, 0.0], H, I, I, [1.0, 0.0])
    assert abs(sum(a) - 1.0) < 1e-12
    assert a[0] > a[1]


def test_narm_bilinear_score_matches_by_hand():
    bs = narm.bilinear_scores([[1.0, 0.0], [0.0, 1.0]],
                              [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                              [2.0, 5.0, 9.0])
    assert abs(bs["scores"][0] - 2.0) < 1e-12
    assert abs(bs["scores"][1] - 5.0) < 1e-12


def test_narm_session_repr_concatenates():
    assert narm.session_repr([1.0, 2.0], [3.0]) == [1.0, 2.0, 3.0]


# ------------------------------------------------------------- fairRC
def test_fairRC_extremes():
    assert fairRC.rND([1, 0] * 25)["value"] < 0.05
    assert abs(fairRC.rND([0] * 40 + [1] * 10)["value"] - 1.0) < 1e-9


def test_fairRC_rRD_flags_the_majority_case():
    r = fairRC.rRD([1] * 40 + [0] * 10)
    assert "caveat" in r


def test_fairRC_rejects_single_group():
    with pytest.raises(ValueError):
        fairRC.rND([1] * 20)
    with pytest.raises(ValueError):
        fairRC.rND([0, 1], step=10)


# ------------------------------------------------------------- node2v
def test_node2v_alpha_values():
    assert node2v.alpha_pq(0, 4.0, 0.25) == 0.25
    assert node2v.alpha_pq(1, 4.0, 0.25) == 1.0
    assert node2v.alpha_pq(2, 4.0, 0.25) == 4.0
    with pytest.raises(ValueError):
        node2v.alpha_pq(3, 1.0, 1.0)


def test_node2v_small_p_backtracks_more():
    def rate(p):
        rng = np.random.default_rng(1)
        back = tot = 0
        for _ in range(300):
            w = node2v.walk(ADJ, 0, 6, p, 1.0, rng)
            for t in range(2, len(w)):
                tot += 1
                back += 1 if w[t] == w[t - 2] else 0
        return back / float(tot)

    assert rate(0.25) > rate(4.0) + 0.05


def test_node2v_walks_follow_edges():
    w = node2v.generate_walks(ADJ, 3, 5, 1.0, 1.0, seed=0)
    for path in w["walks"]:
        for i in range(len(path) - 1):
            assert path[i + 1] in ADJ[path[i]]


# ----------------------------------------------------------- gsageemd
def test_gsageemd_aggregators():
    V = [[1.0, 4.0], [3.0, 0.0], [-1.0, 2.0]]
    assert gsage.aggregate(V, "mean") == [1.0, 2.0]
    assert gsage.aggregate(V, "max_pool") == [3.0, 4.0]
    assert gsage.aggregate(V, "lstm_order") != \
        gsage.aggregate(list(reversed(V)), "lstm_order")


def test_gsageemd_is_inductive():
    W = [[0.5] * 4, [-0.3, 0.2, 0.1, 0.4]]
    f1 = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.8]]
    f2 = [[0.9, 0.1]] + f1[1:]
    a = gsage.embed(f1, ADJ, [W])["embeddings"]
    b = gsage.embed(f2, ADJ, [W])["embeddings"]
    assert a[0] != b[0]


def test_gsageemd_rejects_bad_input():
    with pytest.raises(ValueError):
        gsage.aggregate([[1.0]], "attention")
    with pytest.raises(ValueError):
        gsage.aggregate([], "mean")


# --------------------------------------------------------------- mpfn
EF = {(0, 1): 1.0, (0, 2): 0.5, (1, 2): 2.0, (1, 3): 1.0,
      (2, 4): 1.5, (3, 4): 0.5}
H0 = [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [0.5, 0.5], [1.0, 1.0]]


def test_mpfn_sum_readout_is_permutation_invariant():
    r = mpfn.is_permutation_invariant(H0, ADJ, EF, [2, 0, 4, 1, 3],
                                      T=2, how="sum")
    assert r["invariant"]


def test_mpfn_readout_rejects_unknown_mode():
    with pytest.raises(ValueError):
        mpfn.readout(H0, "first")
    with pytest.raises(ValueError):
        mpfn.readout(H0, "gated")


def test_mpfn_edge_features_change_the_message():
    a = mpfn.message([1.0], [2.0], 1.0)
    b = mpfn.message([1.0], [2.0], 3.0)
    assert a != b


# --------------------------------------------------------- egnnL/egcn
def _phi_e(hi, hj, d2, a):
    return [math.tanh(hi[0] + hj[0] + 0.1 * d2)]


def _phi_x(m):
    return 0.1 * m[0]


def _phi_h(h, m):
    return [math.tanh(h[0] + m[0])]


HS = [[0.5], [-0.2], [0.9]]
XS = [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]


def test_egnnL_is_rotation_and_translation_equivariant():
    th = 0.7
    Q = [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]
    r = egnnL.equivariance_error(HS, XS, _phi_e, _phi_x, _phi_h, Q,
                                 [2.0, -3.0], layers=3)
    assert r["equivariant"]
    assert r["invariant"]


def test_egnnL_is_reflection_equivariant_too():
    r = egnnL.equivariance_error(HS, XS, _phi_e, _phi_x, _phi_h,
                                 [[-1.0, 0.0], [0.0, 1.0]],
                                 [0.0, 0.0], layers=3)
    assert r["equivariant"]


def test_egcn_reexports_egnnL():
    assert egcn.run_egnn is egnnL.run_egnn


def test_egnnL_rejects_bad_input():
    with pytest.raises(ValueError):
        egnnL.coord_update([[1.0]], [[None]], _phi_x)
    with pytest.raises(ValueError):
        egnnL.egcl(HS, XS, _phi_e, _phi_x, _phi_h, mode="velocity")


# -------------------------------------------------------------- gnnEx
KEY = [(0, 1), (1, 2)]


def _predict(edges, em, fm):
    s = sum(em[i] for i in range(len(edges)) if edges[i] in KEY)
    s -= 0.2 * sum(em[i] for i in range(len(edges))
                   if edges[i] not in KEY)
    s += 0.5 * fm[0]
    p = 1.0 / (1.0 + math.exp(-2.0 * (s - 1.0)))
    return [1.0 - p, p]


def test_gnnEx_recovers_the_planted_edges():
    r = gnnEx.explain_node(_predict, ADJ, 1, 1, 3, L=2, iters=250,
                           lr=0.5, size_coef=0.15, entropy_coef=0.05)
    assert all(e in KEY for e, _ in r["edges_ranked"][:2])


def test_gnnEx_penalties_keep_the_mask_small():
    a = gnnEx.explain_node(_predict, ADJ, 1, 1, 3, L=2, iters=250,
                           lr=0.5, size_coef=0.15, entropy_coef=0.05)
    b = gnnEx.explain_node(_predict, ADJ, 1, 1, 3, L=2, iters=250,
                           lr=0.5, penalize=False)
    assert sum(b["edge_mask"]) > sum(a["edge_mask"])


def test_gnnEx_conditional_entropy():
    assert gnnEx.conditional_entropy([0.99, 0.01]) < \
        gnnEx.conditional_entropy([0.5, 0.5])
    assert abs(gnnEx.conditional_entropy([0.5, 0.5])
               - math.log(2)) < 1e-12


def test_gnnEx_computation_graph_grows_with_hops():
    a = gnnEx.computation_graph(ADJ, 0, 1)
    b = gnnEx.computation_graph(ADJ, 0, 2)
    assert len(b["nodes"]) >= len(a["nodes"])


# -------------------------------------------------------------- grace
U = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]


def test_grace_agreeing_views_score_lower():
    good = grace.grace_objective(U, [[0.99, 0.01], [0.01, 0.99],
                                     [1.0, 1.02]])["loss"]
    bad = grace.grace_objective(U, [[0.0, 1.0], [1.0, 0.0],
                                    [-1.0, -1.0]])["loss"]
    assert good < bad - 0.1


def test_grace_intra_view_negatives_matter():
    V = [[0.99, 0.01], [0.01, 0.99], [1.0, 1.02]]
    assert abs(grace.grace_objective(U, V, intra=True)["loss"]
               - grace.grace_objective(U, V,
                                       intra=False)["loss"]) > 1e-6


def test_grace_masks_whole_dimensions():
    rng = np.random.default_rng(3)
    m = grace.mask_features([[1.0] * 10 for _ in range(4)], 0.5, rng)
    for f in range(10):
        assert len(set(m["X"][i][f] for i in range(4))) == 1


def test_grace_rejects_bad_input():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        grace.mask_features([[1.0]], 1.0, rng)
    with pytest.raises(ValueError):
        grace.grace_objective([[1.0]], [[1.0]])
    with pytest.raises(ValueError):
        grace.grace_objective(U, U, tau=0.0)


# --------------------------------------------------------------- gtrf
def test_gtrf_path_graph_encoding_is_a_sinusoid():
    path = {i: [j for j in (i - 1, i + 1) if 0 <= j < 12]
            for i in range(12)}
    pe = gtrf.laplacian_positional_encoding(
        path, 12, dim=2, normalized=False)["encoding"]
    v = [pe[i][0] for i in range(12)]
    ref = [math.cos(math.pi * (i + 0.5) / 12) for i in range(12)]
    num = sum(v[i] * ref[i] for i in range(12))
    den = math.sqrt(sum(q * q for q in v)) * \
        math.sqrt(sum(q * q for q in ref))
    assert abs(abs(num / den) - 1.0) < 1e-6


def test_gtrf_attention_covers_every_node():
    I = [[1.0, 0.0], [0.0, 1.0]]
    H = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.8]]
    out = gtrf.sparse_attention(H, ADJ, I, I, I)["output"]
    assert len(out) == 5


def test_gtrf_rejects_bad_norm():
    I = [[1.0, 0.0], [0.0, 1.0]]
    H = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.8]]
    with pytest.raises(ValueError):
        gtrf.graph_transformer_layer(H, ADJ, I, I, I, I, I,
                                     norm="group")


# -------------------------------------------------------------- meglt
def test_meglt_nuclear_norm():
    A = [[1.0, 2.0], [3.0, 4.0]]
    _, s, _ = np.linalg.svd(A, full_matrices=False)
    assert abs(meglt.nuclear_norm(A) - sum(s)) < 1e-12


def test_meglt_coherence_flags_the_spiked_matrix():
    spike = [[1.0 if (i == 0 and j == 0) else 0.0 for j in range(8)]
             for i in range(8)]
    spread = [[math.cos(i + j) for j in range(8)] for i in range(8)]
    assert meglt.coherence(spike)["mu"] > \
        4.0 * meglt.coherence(spread)["mu"]


def test_meglt_svt_recovers_a_rank_one_matrix():
    u = [1.0, 2.0, -1.0, 0.5, 1.5, -0.5]
    v = [0.5, -1.0, 2.0, 1.0, -0.5, 1.5]
    M = [[u[i] * v[j] for j in range(6)] for i in range(6)]
    rng = np.random.default_rng(7)
    obs = [(i, j) for i in range(6) for j in range(6)
           if float(rng.uniform()) < 0.7]
    r = meglt.svt(M, obs, step=1.2, iters=600)
    assert meglt.relative_error(r["X"], M) < 0.05


def test_meglt_sample_bound_exponents():
    assert meglt.sample_bound(1000, 5, exponent=1.25)["m"] > \
        meglt.sample_bound(1000, 5, exponent=1.2)["m"]
    with pytest.raises(ValueError):
        meglt.sample_bound(1000, 5, exponent=1.5)


# ------------------------------------------------------------- polyak
def test_polyak_halflife_closed_form():
    lh = polyak.lag_halflife(0.001)
    assert abs(lh["halflife"] - math.log(0.5) / math.log(0.999)) < 1e-9


def test_polyak_soft_update_converges_geometrically():
    t = [0.0]
    for _ in range(693):
        t = polyak.soft_update(t, [1.0], 0.001)
    assert abs(t[0] - 0.5) < 0.01


def test_polyak_averaging_beats_the_last_iterate():
    rng = np.random.default_rng(5)
    its, th = [], 3.0
    for t in range(1, 3001):
        th = th - (1.0 / t ** 0.7) * ((th - 1.0)
                                      + 2.0 * (float(rng.uniform())
                                               - 0.5))
        its.append([th])
    av = polyak.polyak_average(its, burn_in=500)["average"][0]
    assert abs(av - 1.0) < abs(its[-1][0] - 1.0)


def test_polyak_hard_update_copies_every_C():
    assert polyak.hard_update([0.0], [1.0], 100, 100)["copied"]
    assert not polyak.hard_update([0.0], [1.0], 101, 100)["copied"]


def test_polyak_rejects_bad_input():
    with pytest.raises(ValueError):
        polyak.soft_update([0.0], [1.0], 0.0)
    with pytest.raises(ValueError):
        polyak.polyak_average([[1.0]], burn_in=5)


# ---------------------------------------------------------------- lsa
X = [[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0],
     [0.0, 0.0, 0.0, 1.0]]


def test_lsa_full_rank_is_plain_term_matching():
    m = lsa.lsa_decompose(X, None, "raw")
    rec = lsa.reconstruct(m)
    assert max(abs(rec[i][j] - X[i][j]) for i in range(5)
               for j in range(4)) < 1e-9


def test_lsa_truncation_retrieves_beyond_the_literal_term():
    m = lsa.lsa_decompose(X, 2, "raw")
    q = lsa.fold_in([1.0, 0.0, 0.0, 0.0, 0.0], m)
    top = [j for j, _ in lsa.cosine_ranking(q, m, 4)["ranking"]]
    assert 2 in top[:3]
    assert X[0][2] == 0.0


def test_lsa_weighting_schemes_differ():
    a = lsa.term_weighting(X, "raw")
    b = lsa.term_weighting(X, "log_entropy")
    assert a != b


def test_lsa_rejects_bad_input():
    with pytest.raises(ValueError):
        lsa.lsa_decompose(X, 99)
    with pytest.raises(ValueError):
        lsa.term_weighting(X, "bm25")


# ------------------------------------------------------------- prsPEG
def test_prsPEG_prioritised_choice_commits():
    g = peg.choice(peg.lit("a"), peg.lit("ab"))
    assert not peg.parse(g, "ab")["matched"]
    assert peg.parse(g, "a")["matched"]


def test_prsPEG_order_changes_the_language():
    assert peg.parse(peg.choice(peg.lit("ab"), peg.lit("a")),
                     "ab")["matched"]


def test_prsPEG_star_is_greedy():
    assert peg.parse(peg.seq(peg.star(peg.lit("a")), peg.lit("b")),
                     "aaab")["matched"]
    assert not peg.parse(peg.seq(peg.star(peg.lit("a")),
                                 peg.lit("a")), "aaa")["matched"]


def test_prsPEG_predicates_do_not_consume():
    assert peg.parse(peg.seq(peg.not_(peg.lit("x")), peg.lit("a")),
                     "a")["matched"]
    assert peg.parse(peg.seq(peg.and_(peg.lit("a")), peg.lit("a")),
                     "a")["matched"]


def test_prsPEG_packrat_memoises():
    r = peg.packrat_parse(peg.seq(peg.star(peg.lit("a")),
                                  peg.lit("b")), "aaaaab")
    assert r["matched"]
    assert r["memo_entries"] > 0


# --------------------------------------------------------------- dqnv
P = [[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]
RW = [[1.0, 0.0], [0.0, 1.0]]


def test_dqnv_converges_to_the_bellman_fixed_point():
    fit = dqnv.q_learning(P, RW, 2, 2, gamma=0.9, alpha=0.2,
                          steps=8000, C=50, seed=1)
    assert fit["final_residual"] < 0.05


def test_dqnv_reward_clipping():
    assert dqnv.clip_reward(500.0) == 1.0
    assert dqnv.clip_reward(-7.0) == -1.0
    assert dqnv.clip_reward(0.3) == 0.3


def test_dqnv_target_uses_the_frozen_copy():
    Qt = [[0.0, 0.0], [5.0, 5.0]]
    assert abs(dqnv.td_target(1.0, 1, Qt, 0.9) - 5.5) < 1e-12
    assert dqnv.td_target(1.0, 1, Qt, 0.9, done=True) == 1.0


def test_dqnv_replay_buffer_is_finite_and_uniform():
    b = dqnv.ReplayBuffer(3)
    for i in range(5):
        b.add(i, 0, 1.0, i + 1)
    assert len(b) == 3
    assert b.data[0][0] == 2
    rng = np.random.default_rng(0)
    assert len(b.sample(2, rng)) == 2


def test_dqnv_rejects_bad_input():
    with pytest.raises(ValueError):
        dqnv.ReplayBuffer(0)
    with pytest.raises(ValueError):
        dqnv.ReplayBuffer(2).sample(1, np.random.default_rng(0))


# ------------------------------------------------------------- resnxt
WINS = [[[0.5, 0.1, 0.0, 0.2]], [[0.0, 0.3, 0.4, 0.1]]]
WMIDS = [[[0.7]], [[-0.2]]]
WOUTS = [[[0.3], [0.1], [0.0], [0.2]],
         [[0.1], [-0.4], [0.2], [0.0]]]


def test_resnxt_block_forms_are_equivalent():
    r = resnxt.block_equivalence([1.0, 0.5, -0.5, 2.0], WINS, WMIDS,
                                 WOUTS)
    assert r["equivalent"]


def test_resnxt_cardinality_trades_against_width():
    a = resnxt.match_complexity(256, 32, 70000)
    b = resnxt.match_complexity(256, 4, 70000)
    assert a["rounded"] < b["rounded"]
    assert abs(a["parameters"] - b["parameters"]) < 0.15 * 70000


def test_resnxt_rejects_bad_input():
    with pytest.raises(ValueError):
        resnxt.block_parameters(0, 4, 4)


# ------------------------------------------------------------- mienco
def test_mienco_jsd_is_bounded_where_dv_is_not():
    assert abs(mienco.jsd_estimate([50.0] * 5, [-50.0] * 5)) < 1.0
    assert mienco.dv_estimate([50.0] * 5, [-50.0] * 5) > 90.0


def test_mienco_matched_patches_score_higher():
    def critic(s, p):
        return sum(s[i] * p[i] for i in range(len(s)))

    good = mienco.local_objective([1.0, 0.0],
                                  [[1.0, 0.1], [0.9, 0.0]],
                                  [[-1.0, 0.0], [-0.9, 0.1]],
                                  critic)["estimate"]
    bad = mienco.local_objective([1.0, 0.0], [[-1.0, 0.0]],
                                 [[1.0, 0.0]], critic)["estimate"]
    assert good > bad


def test_mienco_softplus_closed_form():
    assert abs(mienco.softplus(0.0) - math.log(2.0)) < 1e-15
    assert mienco.softplus(800.0) > 799.0


def test_mienco_rejects_bad_input():
    def critic(s, p):
        return 0.0

    with pytest.raises(ValueError):
        mienco.jsd_estimate([], [1.0])
    with pytest.raises(ValueError):
        mienco.local_objective([1.0], [[1.0]], [[1.0]], critic,
                               estimator="mine")


# -------------------------------------------------------------- pratt
def test_pratt_attention_is_a_distribution():
    H = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    W = [[3.0, 0.0], [0.0, 3.0]]
    r = pratt.sentence_vector(H, W, [0.0, 0.0], [1.0, 0.0])
    assert abs(sum(r["alpha"]) - 1.0) < 1e-12
    assert r["alpha"][0] > r["alpha"][1]


def test_pratt_entropy_extremes():
    assert abs(pratt.attention_entropy([1.0, 0.0, 0.0])
               ["entropy"]) < 1e-9
    assert abs(pratt.attention_entropy([1 / 3.0] * 3)["entropy"]
               - math.log(3)) < 1e-9


def test_pratt_rejects_bad_input():
    with pytest.raises(ValueError):
        pratt.attention([], [[1.0]], [0.0], [1.0])
    with pytest.raises(ValueError):
        pratt.attention_entropy([0.0, 0.0])


def test_batch4_cheatsheets_are_present():
    for mod in (impFB, fmFM, ffmFM, ncfRS, ngcf, gru4r, narm, fairRC,
                node2v, gsage, mpfn, egnnL, egcn, gnnEx, grace, gtrf,
                meglt, polyak, lsa, peg, dqnv, resnxt, mienco, pratt):
        assert len(mod.cheatsheet()) > 80
