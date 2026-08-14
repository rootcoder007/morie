# -*- coding: utf-8 -*-
"""Tests: the 5 TMLE rows and the 10 deep-learning modules."""
import importlib
import math

import pytest

tmlcou = importlib.import_module("morie.fn.tmlcou")
tmlcmp = importlib.import_module("morie.fn.tmlcmp")
tmldgp = importlib.import_module("morie.fn.tmldgp")
tmldyk = importlib.import_module("morie.fn.tmldyk")
tmlcll = importlib.import_module("morie.fn.tmlcll")
unetbk = importlib.import_module("morie.fn.unetbk")
masrcn = importlib.import_module("morie.fn.masrcn")
schN = importlib.import_module("morie.fn.schN")
t5enc = importlib.import_module("morie.fn.t5enc")
sasRec = importlib.import_module("morie.fn.sasRec")
xdeep = importlib.import_module("morie.fn.xdeep")
dits16 = importlib.import_module("morie.fn.dits16")
sortP = importlib.import_module("morie.fn.sortP")
dimNet = importlib.import_module("morie.fn.dimNet")
painn = importlib.import_module("morie.fn.painn")
np = importlib.import_module("morie.fn._array_core")


def _count_data(n=800, seed=3):
    rng = np.random.default_rng(seed)
    A, W, Y, Q1, Q0, g = [], [], [], [], [], []
    for _ in range(n):
        w = float(rng.uniform())
        p = 0.3 + 0.4 * w
        a = 1.0 if float(rng.uniform()) < p else 0.0
        m1, m0 = 2.0 + 3.0 * w, 1.0 + 2.0 * w
        lam = m1 if a == 1.0 else m0
        A.append(a)
        W.append([w])
        Y.append(max(float(int(lam + 2.0 * (float(rng.uniform())
                                            - 0.5))), 0.0))
        Q1.append(m1)
        Q0.append(m0)
        g.append(p)
    return A, W, Y, Q1, Q0, g


# ------------------------------------------------------------- tmlcou
def test_tmlcou_stays_in_range_and_recovers_the_contrast():
    A, W, Y, Q1, Q0, g = _count_data()
    sc = tmlcou.rescale(Y)
    q1 = [(v - sc["lower"]) / sc["range"] for v in Q1]
    q0 = [(v - sc["lower"]) / sc["range"] for v in Q0]
    fit = tmlcou.tmle_count_outcome(Y, A, W, None, g, q1, q0)
    assert fit["in_range"]
    assert abs(fit["psi"] - 1.5) < 0.4
    assert fit["solves_eic"]


def test_tmlcou_linear_fluctuation_escapes_the_range():
    A, W, Y, Q1, Q0, g = _count_data()
    sc = tmlcou.rescale(Y)
    n = len(A)
    H = [A[i] / g[i] - (1 - A[i]) / (1 - g[i]) for i in range(n)]
    q1 = [(v - sc["lower"]) / sc["range"] for v in Q1]
    q0 = [(v - sc["lower"]) / sc["range"] for v in Q0]
    qa = [q1[i] if A[i] == 1.0 else q0[i] for i in range(n)]
    r = tmlcou.linear_fluctuation_unsafe(qa, H, sc["scaled"])
    assert r["out_of_range"] > 0


def test_tmlcou_rescale_is_invertible():
    r = tmlcou.rescale([2.0, 4.0, 6.0])
    assert r["scaled"] == [0.0, 0.5, 1.0]
    assert abs(tmlcou.unscale(0.5, r["lower"], r["upper"])
               - 4.0) < 1e-12


def test_tmlcou_rejects_bad_input():
    with pytest.raises(ValueError):
        tmlcou.tmle_count_outcome([-1.0, 2.0], [1.0, 0.0],
                                  [[0.0], [1.0]])
    with pytest.raises(ValueError):
        tmlcou.rescale([1.0, 1.0])
    with pytest.raises(ValueError):
        tmlcou.rescale([1.0, 5.0], lower=2.0, upper=3.0)


# ------------------------------------------------------------- tmlcmp
TIMES = [1.0, 2.0, 3.0]
H_BASE = {1: [0.2, 0.2, 0.2], 2: [0.05, 0.05, 0.05]}
H_COMP = {1: [0.2, 0.2, 0.2], 2: [0.40, 0.40, 0.40]}


def test_tmlcmp_competing_hazard_lowers_incidence():
    """lambda_1 is identical in both; only the competing risk moves."""
    a = tmlcmp.cumulative_incidence(H_BASE, TIMES)
    b = tmlcmp.cumulative_incidence(H_COMP, TIMES)
    assert b["F"][1][-1] < a["F"][1][-1] - 0.02
    assert H_BASE[1] == H_COMP[1]


def test_tmlcmp_incidences_and_survival_close():
    ci = tmlcmp.cumulative_incidence(H_COMP, TIMES)
    assert max(abs(v - 1.0) for v in ci["closure"]) < 1e-12


def test_tmlcmp_one_minus_km_overstates():
    ci = tmlcmp.cumulative_incidence(H_COMP, TIMES)
    km = tmlcmp.one_minus_km(H_COMP, TIMES, 1)
    assert km["estimate"][-1] > ci["F"][1][-1] + 0.05


def test_tmlcmp_rejects_bad_input():
    with pytest.raises(ValueError):
        tmlcmp.cumulative_incidence({}, TIMES)
    with pytest.raises(ValueError):
        tmlcmp.one_minus_km(H_COMP, TIMES, 99)
    with pytest.raises(ValueError):
        tmlcmp.cause_specific_hazards([1.0], [1, 2], TIMES)


# ------------------------------------------------------------- tmldgp
def _sparse(n=400, seed=21):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        row = [float(rng.uniform()) - 0.5 for _ in range(12)]
        X.append(row)
        y.append(2.0 * row[0] - 1.5 * row[1]
                 + 0.3 * (float(rng.uniform()) - 0.5))
    return X, y


def test_tmldgp_lasso_selects_and_post_lasso_unshrinks():
    X, y = _sparse()
    las = tmldgp.lasso_path(X, y, 0.05)
    pl = tmldgp.post_lasso(X, y, 0.05)
    assert set([0, 1]) <= set(las["support"])
    assert abs(pl["coef"][1] - 2.0) < abs(las["beta"][0] - 2.0)


def test_tmldgp_penalised_nuisances_still_solve_the_score():
    rng = np.random.default_rng(22)
    A, W, Y = [], [], []
    for _ in range(500):
        w = [float(rng.uniform()) - 0.5 for _ in range(6)]
        p = min(max(0.5 + 0.6 * w[0], 0.1), 0.9)
        a = 1.0 if float(rng.uniform()) < p else 0.0
        m = min(max(0.3 + 0.3 * a + 0.3 * w[0], 0.02), 0.98)
        A.append(a)
        W.append(w)
        Y.append(1.0 if float(rng.uniform()) < m else 0.0)
    assert tmldgp.penalised_tmle(Y, A, W, 0.02)["solves_eic"]


def test_tmldgp_penalising_the_fluctuation_breaks_it():
    r = tmldgp.shrunk_targeting_unsafe([0.4] * 200, [1.0] * 200,
                                       [1.0] * 60 + [0.0] * 140,
                                       ridge=50.0)
    assert abs(r["score"]) > 1e-3


def test_tmldgp_rejects_bad_input():
    X, y = _sparse(50)
    with pytest.raises(ValueError):
        tmldgp.lasso_path(X, y, -1.0)
    with pytest.raises(ValueError):
        tmldgp.penalised_tmle([2.0, 0.0], [1.0, 0.0],
                              [[0.0], [1.0]])


# ------------------------------------------------------------- tmldyk
def test_tmldyk_sensitivity_carries_one_over_g():
    a = tmldyk.ate_sensitivity(1000, 0.5)
    b = tmldyk.ate_sensitivity(1000, 0.02)
    assert abs(b["sensitivity"] / a["sensitivity"] - 25.0) < 1e-9


def test_tmldyk_laplace_variance_is_two_b_squared():
    rng = np.random.default_rng(5)
    d = [tmldyk.laplace_noise(1.0, rng) for _ in range(20000)]
    assert abs(sum(v * v for v in d) / len(d) - 2.0) < 0.15


def test_tmldyk_smaller_epsilon_more_noise():
    a = tmldyk.private_release(0.3, 0.01, 0.1, seed=1)
    b = tmldyk.private_release(0.3, 0.01, 10.0, seed=1)
    assert abs(a["noise"]) > abs(b["noise"])


def test_tmldyk_private_interval_is_wider():
    r = tmldyk.private_ci(0.3, 0.01, 0.5, 0.02, seed=2)
    assert r["width_ratio"] > 1.0


def test_tmldyk_composition_adds():
    assert tmldyk.composition_budget([0.1, 0.2,
                                      0.3])["total_epsilon"] == 0.6


def test_tmldyk_rejects_bad_input():
    with pytest.raises(ValueError):
        tmldyk.ate_sensitivity(100, 0.9)
    with pytest.raises(ValueError):
        tmldyk.private_release(1.0, 0.1, 0.0)
    with pytest.raises(ValueError):
        tmldyk.composition_budget([0.1, -0.2])


# ------------------------------------------------------------- tmlcll
def _panel(npers=300, T=4, seed=31):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(npers):
        u = 3.0 * (float(rng.uniform()) - 0.5)
        xs, ys, px, py = [], [], 0.0, 0.0
        for _ in range(T):
            px = u + 0.3 * px + (float(rng.uniform()) - 0.5)
            py = u + 0.3 * py + (float(rng.uniform()) - 0.5)
            xs.append(px)
            ys.append(py)
        X.append(xs)
        Y.append(ys)
    return X, Y


def test_tmlcll_random_intercept_removes_the_spurious_cross_lag():
    X, Y = _panel()
    cl = tmlcll.clpm_coefficients(X, Y)
    ri = tmlcll.ri_clpm_coefficients(X, Y)
    assert abs(cl["cross_lag_x_to_y"]) > 0.15
    assert abs(ri["cross_lag_x_to_y"]) < 0.10


def test_tmlcll_decomposition_separates_the_variances():
    X, _ = _panel()
    d = tmlcll.within_between_decomposition(X)
    assert d["between_variance"] > d["within_variance"]


def test_tmlcll_targeted_contrast_solves_the_score():
    rng = np.random.default_rng(33)
    A, W, Y, T = [], [], [], []
    for i in range(400):
        w = float(rng.uniform())
        a = 1.0 if float(rng.uniform()) < 0.3 + 0.4 * w else 0.0
        m = min(max(0.2 + 0.3 * a + 0.3 * w, 0.02), 0.98)
        A.append(a)
        W.append([w])
        Y.append(1.0 if float(rng.uniform()) < m else 0.0)
        T.append(i % 4)
    r = tmlcll.tmle_cross_lagged(Y, A, W, T, bounds=(0.0, 1.0))
    assert r["solves_eic"]
    assert abs(r["psi"] - 0.3) < 0.15


def test_tmlcll_rejects_bad_input():
    with pytest.raises(ValueError):
        tmlcll.clpm_coefficients([[1.0]], [[1.0]])
    with pytest.raises(ValueError):
        tmlcll.clpm_coefficients([[1.0, 2.0]], [[1.0]])


# ------------------------------------------------------------- unetbk
def test_unetbk_published_arithmetic():
    assert unetbk.valid_output_size(572, depth=4)["output"] == 388


def test_unetbk_mirrors_the_border():
    img = [[float(i * 3 + j) for j in range(3)] for i in range(3)]
    mp = unetbk.mirror_pad(img, 1)
    assert mp[0][1] == img[1][0]
    assert mp[1][0] == img[0][1]


def test_unetbk_tiles_abut_on_output():
    r = unetbk.overlap_tiles(400, 400, 200, 50)
    assert r["output_size"] == 100
    assert r["n_tiles"] == 16


def test_unetbk_skip_connection_centre_crops():
    r = unetbk.skip_concat([[1.0, 2.0]],
                           [[0.0, 0.0, 0.0, 0.0],
                            [0.0, 9.0, 9.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]])
    assert r["crop_offset"] == (1, 1)


def test_unetbk_weight_map_favours_the_gap():
    lab = [[0, 0, 0, 0, 0], [0, 1, 0, 2, 0], [0, 0, 0, 0, 0]]
    w = unetbk.separation_weight_map(lab)
    assert w["weights"][1][2] > w["weights"][0][0]


def test_unetbk_rejects_bad_input():
    with pytest.raises(ValueError):
        unetbk.mirror_pad([[1.0]], 3)
    with pytest.raises(ValueError):
        unetbk.overlap_tiles(100, 100, 10, 10)
    with pytest.raises(ValueError):
        unetbk.valid_output_size(4, depth=4)


# ------------------------------------------------------------- masrcn
FEAT = [[float(i * 8 + j) for j in range(8)] for i in range(8)]
BOX = (21.0, 27.0, 85.0, 91.0)


def test_masrcn_roipool_quantisation_is_pixels():
    r = masrcn.alignment_error(FEAT, BOX, 2, stride=16.0)
    assert max(r["input_pixel_shift"]) > 4.0


def test_masrcn_align_and_pool_disagree():
    a = masrcn.roi_pool(FEAT, BOX, 2, stride=16.0)["pooled"]
    b = masrcn.roi_align(FEAT, BOX, 2, stride=16.0)["pooled"]
    assert max(abs(a[i][j] - b[i][j])
               for i in range(2) for j in range(2)) > 0.5


def test_masrcn_sigmoid_and_softmax_are_different_losses():
    lg = [[3.0, -3.0], [-3.0, 3.0]]
    tg = [[1.0, 0.0], [0.0, 1.0]]
    assert abs(masrcn.mask_loss(lg, tg, True)["loss"]
               - masrcn.mask_loss(lg, tg, False)["loss"]) > 0.1


def test_masrcn_multitask_loss_is_a_sum():
    assert abs(masrcn.multitask_loss(0.1, 0.2, 0.3)["total"]
               - 0.6) < 1e-12


def test_masrcn_rejects_a_degenerate_box():
    with pytest.raises(ValueError):
        masrcn.roi_align(FEAT, (1.0, 1.0, 1.0, 1.0), 2)


# --------------------------------------------------------------- schN
def _energy(pos):
    e = 0.0
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            d = math.sqrt(sum((pos[i][a] - pos[j][a]) ** 2
                              for a in range(3)))
            e += math.exp(-d) * schN.cosine_cutoff(d, 5.0)
    return e


R3 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.2, 0.0]]


def test_schN_energy_invariant_forces_equivariant():
    th = 0.6
    Q = [[math.cos(th), -math.sin(th), 0.0],
         [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]]
    r = schN.invariance_error(_energy, R3, Q, [1.0, -2.0, 0.5])
    assert r["energy_invariant"]
    assert r["forces_equivariant"]


def test_schN_forces_sum_to_zero():
    r = schN.forces_from_energy(_energy, R3)
    assert max(abs(v) for v in r["net_force"]) < 1e-6


def test_schN_cutoff_is_continuous():
    assert abs(schN.cosine_cutoff(4.9999, 5.0)) < 1e-6
    assert schN.cosine_cutoff(5.1, 5.0) == 0.0
    assert abs(schN.cosine_cutoff(0.0, 5.0) - 1.0) < 1e-12


def test_schN_rejects_bad_input():
    with pytest.raises(ValueError):
        schN.gaussian_expansion(1.0, n_gaussians=1)
    with pytest.raises(ValueError):
        schN.cosine_cutoff(1.0, 0.0)


# -------------------------------------------------------------- t5enc
def test_t5enc_span_corruption_shortens_the_target():
    toks = ["a", "b", "c", "d", "e", "f", "g", "h"] * 4
    r = t5enc.span_corruption(toks, 0.15, 3.0, seed=2)
    assert len(r["target"]) < len(r["input"])
    assert r["n_spans"] < r["corrupted_tokens"]


def test_t5enc_relative_buckets_share_at_distance():
    near = [t5enc.relative_bucket(d) for d in (0, 1, 2, 3)]
    far = [t5enc.relative_bucket(d) for d in (100, 110, 120)]
    assert len(set(near)) == 4
    assert len(set(far)) < 3


def test_t5enc_invalid_label_is_wrong():
    assert not t5enc.parse_prediction("purple",
                                      ["yes", "no"])["valid"]
    assert t5enc.parse_prediction("yes", ["yes", "no"])["valid"]
    assert not t5enc.parse_prediction("abc")["valid"]


def test_t5enc_regression_is_rounded_and_clipped():
    assert t5enc.format_regression(3.27) == "3.2"
    assert t5enc.format_regression(9.0, hi=5.0) == "5.0"


def test_t5enc_rejects_bad_input():
    with pytest.raises(ValueError):
        t5enc.task_prefix("  ", "text")
    with pytest.raises(ValueError):
        t5enc.span_corruption(["a", "b"], rate=1.5)
    with pytest.raises(ValueError):
        t5enc.relative_bucket(3, num_buckets=1)


# ------------------------------------------------------------- sasRec
I2 = [[1.0, 0.0], [0.0, 1.0]]


def test_sasRec_causal_mask():
    m = sasRec.causal_mask(4)
    assert all(m[i][j] == 0.0 for i in range(4) for j in range(4)
               if j > i)


def test_sasRec_span_adapts_to_the_signal():
    recent = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [3.0, 0.0]]
    distant = [[3.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    a = sasRec.attention_span(
        sasRec.self_attention(recent, I2, I2, I2)["weights"])
    b = sasRec.attention_span(
        sasRec.self_attention(distant, I2, I2, I2)["weights"])
    assert b["mean_lookback"] > a["mean_lookback"] + 0.5


def test_sasRec_complexity_contrast():
    c = sasRec.complexity(200, 64)
    assert c["attention_sequential_steps"] == 1
    assert c["rnn_sequential_steps"] == 200


def test_sasRec_rejects_bad_input():
    with pytest.raises(ValueError):
        sasRec.causal_mask(0)
    with pytest.raises(ValueError):
        sasRec.attention_span([[0.0, 0.0]], 0)


# -------------------------------------------------------------- xdeep
def test_xdeep_hadamard_is_vector_wise():
    assert xdeep.hadamard([1.0, 2.0], [3.0, 4.0]) == [3.0, 8.0]
    with pytest.raises(ValueError):
        xdeep.hadamard([1.0], [1.0, 2.0])


def test_xdeep_layer_k_holds_degree_k_plus_one():
    X0 = [[1.0, 2.0], [3.0, 1.0]]
    W = [[[1.0, 0.0], [0.0, 0.0]]]
    c = xdeep.cin(X0, [W, W])
    assert c["degrees"] == [2, 3]
    assert c["layers"][0][0] == [1.0, 4.0]


def test_xdeep_degree_is_explicit():
    assert xdeep.interaction_degree(0)["degree"] == 2
    assert xdeep.interaction_degree(3)["degree"] == 5
    with pytest.raises(ValueError):
        xdeep.interaction_degree(-1)


# ------------------------------------------------------------- dits16
def test_dits16_patch_halving_quadruples_tokens():
    assert dits16.patch_grid(32, 2)["tokens"] == \
        4 * dits16.patch_grid(32, 4)["tokens"]


def test_dits16_tokens_move_gflops():
    a = dits16.gflops(dits16.patch_grid(32, 2)["tokens"], 28, 1152)
    b = dits16.gflops(dits16.patch_grid(32, 4)["tokens"], 28, 1152)
    assert a["gflops"] > 4.0 * b["gflops"]


def test_dits16_adaln_zero_is_the_identity_at_init():
    Z = [[0.0, 0.0], [0.0, 0.0]]
    r = dits16.adaln_zero([1.0, 1.0], [1.0, 3.0], Z, Z, Z)
    assert r["identity_at_init"]


def test_dits16_rejects_bad_input():
    with pytest.raises(ValueError):
        dits16.patch_grid(33, 2)
    with pytest.raises(ValueError):
        dits16.gflops(0, 1, 1)


# -------------------------------------------------------------- sortP
FEATS = [[0.1, 0.9], [0.2, 0.3], [0.3, 0.6], [0.4, 0.1]]


def test_sortP_sorts_and_fixes_the_size():
    r = sortP.sort_pooling(FEATS, 3)
    assert r["order"] == [0, 2, 1]
    assert len(r["pooled"]) == 3


def test_sortP_order_is_graph_determined():
    assert sortP.order_is_graph_determined(FEATS, {}, [2, 0, 3, 1],
                                           3)["invariant"]


def test_sortP_pads_short_graphs():
    r = sortP.sort_pooling(FEATS, 6)
    assert r["n_padded"] == 2
    assert r["pooled"][-1] == [0.0, 0.0]


def test_sortP_k_is_a_coverage_quantile():
    r = sortP.choose_k([5, 8, 10, 12, 30], 0.6)
    assert r["k"] == 10
    assert r["fraction_untruncated"] >= 0.6


def test_sortP_wl_colours_reflect_structural_role():
    c = sortP.wl_colours({0: [1], 1: [0, 2], 2: [1]}, 3, rounds=2)
    assert abs(c[0] - c[2]) < 1e-12
    assert abs(c[1] - c[0]) > 1e-9


def test_sortP_rejects_bad_input():
    with pytest.raises(ValueError):
        sortP.sort_pooling(FEATS, 0)
    with pytest.raises(ValueError):
        sortP.choose_k([], 0.5)
    with pytest.raises(ValueError):
        sortP.choose_k([1, 2], 0.0)


# ------------------------------------------------------------- dimNet
def test_dimNet_angle_distinguishes_equal_distances():
    a = dimNet.angle_between([0.0, 0.0], [1.0, 0.0], [2.0, 0.0])
    b = dimNet.angle_between([0.0, 0.0], [1.0, 0.0], [1.0, 1.0])
    assert abs(a - math.pi) < 1e-9
    assert abs(b - math.pi / 2) < 1e-9


def test_dimNet_cost_is_in_triplets():
    r = dimNet.triplet_count({0: [1], 1: [0, 2], 2: [1, 3], 3: [2]})
    assert r["triplets"] == 4
    assert r["pairs"] == 6


def test_dimNet_bessel_basis_is_orthogonal():
    ip = sum(dimNet.bessel_basis(0.01 + 5.0 * i / 4000.0, 5.0, 8)[0]
             * dimNet.bessel_basis(0.01 + 5.0 * i / 4000.0,
                                   5.0, 8)[1]
             * (0.01 + 5.0 * i / 4000.0) ** 2
             for i in range(4000)) * (5.0 / 4000.0)
    assert abs(ip) < 0.02


def test_dimNet_legendre_at_one():
    assert all(abs(v - 1.0) < 1e-12
               for v in dimNet.spherical_harmonic_basis(0.0, 4))


def test_dimNet_rejects_bad_input():
    with pytest.raises(ValueError):
        dimNet.angle_between([0.0, 0.0], [0.0, 0.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        dimNet.bessel_basis(0.0, 5.0)


# -------------------------------------------------------------- painn
def _toy(s, v, R):
    nrm = painn.vector_norm(v)
    return {"s": [s[f] + nrm[f] for f in range(len(s))],
            "v": [[2.0 * v[a][f] for f in range(len(v[0]))]
                  for a in range(len(v))]}


def test_painn_scalars_invariant_vectors_equivariant():
    th = 0.9
    Q = [[math.cos(th), -math.sin(th), 0.0],
         [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]]
    r = painn.equivariance_error(_toy, [0.5, -0.2],
                                 [[1.0, 0.0], [0.0, 2.0],
                                  [0.5, 0.5]],
                                 [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                                 Q)
    assert r["scalars_invariant"]
    assert r["vectors_equivariant"]


def test_painn_norm_is_the_invariant_channel():
    n = painn.vector_norm([[1.0, 0.0], [0.0, 2.0], [0.5, 0.5]])
    assert abs(n[0] - math.sqrt(1.25)) < 1e-12


def test_painn_dipole_is_a_vector_property():
    d = painn.dipole_moment([1.0, -1.0], [[1.0, 0.0, 0.0],
                                          [-1.0, 0.0, 0.0]])
    assert abs(d["magnitude"] - 2.0) < 1e-12
    assert len(d["dipole"]) == 3


def test_painn_rejects_bad_input():
    with pytest.raises(ValueError):
        painn.dipole_moment([1.0], [[0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0]])


def test_tml_dl_cheatsheets_are_present():
    for mod in (tmlcou, tmlcmp, tmldgp, tmldyk, tmlcll, unetbk,
                masrcn, schN, t5enc, sasRec, xdeep, dits16, sortP,
                dimNet, painn):
        assert len(mod.cheatsheet()) > 80
