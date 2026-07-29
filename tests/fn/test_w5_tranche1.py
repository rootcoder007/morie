"""Independent-route checks for the w5 tranche-1 Géron modules.

Every assertion here is derived from the stated formula by a route that
does *not* reuse the implementation: closed forms, hand-derived first
steps, brute-force enumeration, or an algebraic identity.  A
mean-of-inputs placeholder fails all of them.

Deterministic pseudo-randomness comes from the LCG
``s = (1664525*s + 1013904223) % 2**32; u = (s + 0.5) / 2**32`` so the
tests are reproducible without seeding numpy.
"""

import math

import numpy as np
import pytest

from morie.fn.gr1cy import geron_1cycle_schedule
from morie.fn.grac import geron_actor_critic_advantage
from morie.fn.grada2 import geron_adagrad_update
from morie.fn.gradaw import geron_adaboost_weight_update
from morie.fn.gradmo import geron_adam_update
from morie.fn.grael import geron_autoencoder_reconstruction_loss
from morie.fn.graic import geron_aic_gmm, gmm_n_params
from morie.fn.grarma import geron_arima_forecast
from morie.fn.grauc import geron_auc_roc
from morie.fn.graut import geron_autograd_chain_rule
from morie.fn.grbag import geron_bagging_predictor
from morie.fn.grbah import geron_bahdanau_attention
from morie.fn.grbeam import geron_beam_search_decoder
from morie.fn.grbf16 import geron_bf16_range
from morie.fn.grbgd import geron_batch_gradient_descent
from morie.fn.grbic import geron_bic_gmm
from morie.fn.grblip import geron_blip_itm_itc
from morie.fn.grbn import geron_batch_normalization
from morie.fn.grbo import geron_bellman_optimality
from morie.fn.grbp import geron_backpropagation_gradient
from morie.fn.grbpe import geron_bpe_tokenizer_merge
from morie.fn.grbptt import geron_backprop_through_time
from morie.fn.grbrnn import geron_bidirectional_rnn
from morie.fn.grca import geron_cross_attention
from morie.fn.grcae import geron_convolutional_autoencoder
from morie.fn.grcart import geron_cart_split_cost
from morie.fn.grcfm import geron_confusion_matrix
from morie.fn.grclp import geron_clip_contrastive_loss
from morie.fn.grcos import geron_conv_output_size
from morie.fn.grctr import geron_contrastive_infonce
from morie.fn.grcvf import geron_conv2d_forward
from morie.fn.grcvs import geron_cross_validation_score
from morie.fn.grdae import geron_denoising_autoencoder
from morie.fn.grdal import geron_dalle_autoregressive_token
from morie.fn.grdbs import geron_dbscan_core_point
from morie.fn.grdcgan import geron_dcgan_generator
from morie.fn.grddim import geron_ddim_sampling_step
from morie.fn.grddqn import geron_double_dqn_target
from morie.fn.grdeit import geron_deit_distillation_loss
from morie.fn.grdetr import geron_detr_hungarian_matching


# ── deterministic LCG ────────────────────────────────────────────────

def lcg(n, seed=12345):
    """n draws uniform on (0, 1) from the stated LCG."""
    s = seed
    out = []
    for _ in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out.append((s + 0.5) / 2**32)
    return np.asarray(out, dtype=float)


def lcg_normal(n, seed=12345):
    """n standard normals via Box-Muller on LCG uniforms."""
    u = lcg(2 * ((n + 1) // 2), seed)
    u1 = np.clip(u[0::2], 1e-12, 1.0)
    u2 = u[1::2]
    r = np.sqrt(-2.0 * np.log(u1))
    z = np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])
    return z[:n]


# ── gr1cy: 1cycle schedule ───────────────────────────────────────────

def test_gr1cy_closed_form_and_shape():
    T, lo, hi = 11, 0.01, 0.1
    r = geron_1cycle_schedule(lo, hi, t=0, T=T)
    lr = np.asarray(r["lr_schedule"])
    assert lr.size == T                       # full curve, not one point
    peak = r["peak_step"]
    assert peak == (T - 1) // 2
    # Closed form: linear interpolation on [0, peak] and [peak, T-1].
    for i in range(T):
        if i <= peak:
            want = lo + (hi - lo) * i / peak
        else:
            want = hi + (lo - hi) * (i - peak) / (T - 1 - peak)
        assert lr[i] == pytest.approx(want, abs=1e-12)


def test_gr1cy_monotone_up_then_down_and_momentum_mirrors():
    r = geron_1cycle_schedule(0.01, 0.1, t=3, T=21)
    lr = np.asarray(r["lr_schedule"])
    mom = np.asarray(r["momentum_schedule"])
    peak = r["peak_step"]
    assert np.all(np.diff(lr[:peak + 1]) > 0)
    assert np.all(np.diff(lr[peak:]) < 0)
    # Momentum is the mirror: it falls while the LR rises and vice versa.
    assert np.all(np.diff(mom[:peak + 1]) < 0)
    assert np.all(np.diff(mom[peak:]) > 0)
    assert float(r) == pytest.approx(lr[3])


def test_gr1cy_rejects_bad_bounds():
    with pytest.raises(ValueError):
        geron_1cycle_schedule(0.5, 0.1, t=0, T=5)
    with pytest.raises(ValueError):
        geron_1cycle_schedule(0.1, 0.5, t=5, T=5)


# ── grac: actor-critic advantage ─────────────────────────────────────

def test_grac_matches_hand_computed_td_error():
    V = [1.0, 4.0, -2.0]
    s = [0, 1, 2]
    s_next = [1, 2, 0]
    r = [0.5, -1.0, 2.0]
    g = 0.9
    want = [r[k] + g * V[s_next[k]] - V[s[k]] for k in range(3)]
    got = geron_actor_critic_advantage(V, s, s_next, r, g)["advantage"]
    assert got == pytest.approx(want)


def test_grac_terminal_drops_bootstrap():
    r = geron_actor_critic_advantage([5.0, 100.0], [0], [1], [1.0], 0.9, done=[True])
    assert r["advantage"][0] == pytest.approx(1.0 - 5.0)


# ── grada2: AdaGrad ──────────────────────────────────────────────────

def test_grada2_first_step_and_monotone_lr_decay():
    theta, g, s, eta = np.array([1.0, -2.0]), np.array([3.0, 4.0]), np.zeros(2), 0.5
    r = geron_adagrad_update(theta, g, s, eta, eps=0.0)
    assert r["s_new"] == pytest.approx([9.0, 16.0])
    # From s=0 the step is exactly eta*sign(g), independent of |g|.
    assert r["step"] == pytest.approx([0.5, 0.5])
    assert r["theta_new"] == pytest.approx([0.5, -2.5])
    # A second identical gradient must give a strictly smaller step.
    r2 = geron_adagrad_update(r["theta_new"], g, r["s_new"], eta, eps=0.0)
    assert np.all(np.abs(np.asarray(r2["step"])) < np.abs(np.asarray(r["step"])))


def test_grada2_rejects_negative_accumulator():
    with pytest.raises(ValueError):
        geron_adagrad_update([1.0], [1.0], [-1.0], 0.1)


# ── gradaw: AdaBoost weights ─────────────────────────────────────────

def test_gradaw_boost_ratio_is_exp_alpha_after_normalisation():
    alpha = 0.7
    w = [0.2, 0.3, 0.5]
    y, p = [0, 1, 1], [1, 1, 1]          # only sample 0 is wrong
    r = geron_adaboost_weight_update(y, p, w, alpha)
    new = np.asarray(r["weights_new"])
    assert new.sum() == pytest.approx(1.0)
    # Ratio of a wrong to a right sample scales by exactly exp(alpha).
    assert (new[0] / new[1]) == pytest.approx((w[0] / w[1]) * math.exp(alpha))
    assert r["weighted_error"] == pytest.approx(0.2)


def test_gradaw_alpha_zero_only_renormalises():
    r = geron_adaboost_weight_update([0, 1], [1, 1], [1.0, 3.0], 0.0)
    assert r["weights_new"] == pytest.approx([0.25, 0.75])


# ── gradmo: Adam ─────────────────────────────────────────────────────

def test_gradmo_bias_correction_makes_first_step_eta():
    eta = 0.01
    r = geron_adam_update([0.0], [7.0], [0.0], [0.0], t=1, eta=eta,
                          b1=0.9, b2=0.999, eps=1e-12)
    # m_hat = g, s_hat = g^2 at t=1 -> step = eta * g / |g| = eta.
    assert r["m_hat"][0] == pytest.approx(7.0)
    assert r["s_hat"][0] == pytest.approx(49.0)
    assert r["step"][0] == pytest.approx(eta, rel=1e-9)
    # Without bias correction the step would be ~eta*0.1/sqrt(0.001) = 100x off.
    assert abs(r["step"][0] - eta * 0.1 / math.sqrt(0.001)) > 1e-6


def test_gradmo_second_step_matches_hand_recursion():
    b1, b2, eta, eps = 0.9, 0.999, 0.001, 1e-8
    g1, g2 = 2.0, -1.0
    m1 = (1 - b1) * g1
    s1 = (1 - b2) * g1**2
    m2 = b1 * m1 + (1 - b1) * g2
    s2 = b2 * s1 + (1 - b2) * g2**2
    step2 = eta * (m2 / (1 - b1**2)) / (math.sqrt(s2 / (1 - b2**2)) + eps)
    r1 = geron_adam_update([0.0], [g1], [0.0], [0.0], 1, eta, b1, b2, eps)
    r2 = geron_adam_update(r1["theta_new"], [g2], r1["m_new"], r1["s_new"],
                           2, eta, b1, b2, eps)
    assert r2["m_new"][0] == pytest.approx(m2)
    assert r2["s_new"][0] == pytest.approx(s2)
    assert r2["step"][0] == pytest.approx(step2)


def test_gradmo_rejects_zero_step_counter():
    with pytest.raises(ValueError):
        geron_adam_update([1.0], [1.0], [0.0], [0.0], t=0, eta=0.1)


# ── grael: autoencoder loss ──────────────────────────────────────────

def test_grael_loss_is_mean_squared_l2_norm():
    X = lcg_normal(12, seed=7).reshape(4, 3)
    D = X + lcg_normal(12, seed=99).reshape(4, 3) * 0.1
    want = float(np.mean(np.sum((X - D) ** 2, axis=1)))
    r = geron_autoencoder_reconstruction_loss(X, X[:, :1], D)
    assert r["loss"] == pytest.approx(want)
    assert r["compression_ratio"] == pytest.approx(3.0)
    assert r["loss"] != pytest.approx(float(np.mean(X)))    # not a mean-of-inputs stub


# ── graic / grbic ────────────────────────────────────────────────────

def test_graic_grbic_penalty_difference_is_p_times_logn_minus_2():
    ll, p, n = -37.5, 8, 250
    a = geron_aic_gmm(ll, p)["aic"]
    b = geron_bic_gmm(ll, n, p)["bic"]
    assert (b - a) == pytest.approx(p * (math.log(n) - 2.0))
    assert a == pytest.approx(2 * p - 2 * ll)


def test_gmm_n_params_counts_free_parameters():
    # k-1 weights + k*d means + k*d(d+1)/2 covariance entries
    assert gmm_n_params(4, 3, "full") == 3 + 12 + 4 * 6
    assert gmm_n_params(4, 3, "diag") == 3 + 12 + 12
    with pytest.raises(ValueError):
        gmm_n_params(1, 2, "banana")


# ── grarma ───────────────────────────────────────────────────────────

def test_grarma_ar1_forecast_and_integration():
    # w = diff(y) is constant 3, phi=0.5 -> w_hat = 1.5, level = y[-1] + 1.5
    y = [0.0, 3.0, 6.0, 9.0]
    r = geron_arima_forecast(y, [0.5], [], d=1)
    assert r["forecast_differenced"] == pytest.approx(1.5)
    assert r["forecast"] == pytest.approx(10.5)


def test_grarma_ma_residual_recursion_matches_hand_unrolling():
    y = [2.0, 1.0, -1.0]
    th = 0.4
    e0 = 2.0
    e1 = 1.0 - th * e0
    e2 = -1.0 - th * e1
    r = geron_arima_forecast(y, [], [th], d=0)
    assert r["residuals"] == pytest.approx([e0, e1, e2])
    assert r["forecast"] == pytest.approx(th * e2)


def test_grarma_rejects_too_short_series():
    with pytest.raises(ValueError):
        geron_arima_forecast([1.0], [0.5, 0.5], [], d=1)


# ── grauc ────────────────────────────────────────────────────────────

def test_grauc_equals_mann_whitney_with_half_credit_for_ties():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    sc = np.round(lcg(8, seed=3) * 3) / 3.0          # deliberate ties
    pos, neg = sc[y == 1], sc[y == 0]
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    want = wins / (pos.size * neg.size)
    assert geron_auc_roc(y, sc)["auc"] == pytest.approx(want)


def test_grauc_roc_curve_is_monotone_and_anchored():
    y = [0, 1, 0, 1, 1, 0]
    sc = lcg(6, seed=41)
    r = geron_auc_roc(y, sc)
    fpr, tpr = np.asarray(r["fpr"]), np.asarray(r["tpr"])
    assert np.all(np.diff(fpr) >= -1e-12) and np.all(np.diff(tpr) >= -1e-12)
    assert (fpr[0], tpr[0]) == (0.0, 0.0)
    assert (fpr[-1], tpr[-1]) == (1.0, 1.0)


def test_grauc_needs_both_classes():
    with pytest.raises(ValueError):
        geron_auc_roc([1, 1, 1], [0.1, 0.2, 0.3])


# ── graut ────────────────────────────────────────────────────────────

def test_graut_matches_explicit_jacobian_product():
    J1 = lcg_normal(6, seed=5).reshape(3, 2)     # x(2) -> u(3)
    J2 = lcg_normal(3, seed=6).reshape(1, 3)     # u(3) -> y(1)
    g = np.array([2.5])
    want = g @ J2 @ J1
    got = geron_autograd_chain_rule([J1, J2], g)["grad_input"]
    assert got == pytest.approx(want)


def test_graut_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        geron_autograd_chain_rule([np.ones((2, 2))], [1.0, 1.0, 1.0])


# ── grbag ────────────────────────────────────────────────────────────

def test_grbag_mean_and_variance_reduction():
    P = lcg_normal(30, seed=11).reshape(6, 5)
    r = geron_bagging_predictor(P)
    assert r["prediction"] == pytest.approx(P.mean(axis=0))
    assert r["per_instance_variance"] == pytest.approx(P.var(axis=0, ddof=1))
    # SE of the ensemble mean shrinks as 1/sqrt(B).
    assert r["se"] == pytest.approx(math.sqrt(P.var(axis=0, ddof=1).mean() / 6))


def test_grbag_hard_vote_picks_the_mode():
    assert geron_bagging_predictor([[0, 2], [1, 2], [1, 5]],
                                   aggregate="vote")["prediction"] == [1.0, 2.0]


# ── grbah ────────────────────────────────────────────────────────────

def test_grbah_scores_and_context_by_explicit_loop():
    h = lcg_normal(2, seed=13)
    S = lcg_normal(6, seed=14).reshape(3, 2)
    Wh = lcg_normal(4, seed=15).reshape(2, 2)
    Ws = lcg_normal(4, seed=16).reshape(2, 2)
    v = lcg_normal(2, seed=17)
    e = np.array([float(v @ np.tanh(Wh @ h + Ws @ S[i])) for i in range(3)])
    a = np.exp(e - e.max())
    a = a / a.sum()
    r = geron_bahdanau_attention(h, S, Wh, Ws, v)
    assert r["scores"] == pytest.approx(e)
    assert r["weights"] == pytest.approx(a)
    assert r["context"] == pytest.approx(a @ S)
    assert sum(r["weights"]) == pytest.approx(1.0)


# ── grbeam ───────────────────────────────────────────────────────────

def test_grbeam_wide_beam_matches_brute_force():
    S = np.round(lcg_normal(12, seed=19).reshape(4, 3), 3)
    T, V = S.shape
    best, best_score = None, -np.inf
    import itertools
    for seq in itertools.product(range(V), repeat=T):
        sc = sum(S[t, seq[t]] for t in range(T))
        if sc > best_score:
            best, best_score = list(seq), sc
    r = geron_beam_search_decoder(S, beam_width=V**T)
    assert r["best_sequence"] == best
    assert r["best_score"] == pytest.approx(best_score)


def test_grbeam_keeps_exactly_beam_width_hypotheses():
    S = [[-0.1, -0.5, -2.0], [-0.2, -0.4, -3.0]]
    r = geron_beam_search_decoder(S, beam_width=3)
    assert len(r["beams"]) == 3
    scores = [v for _, v in r["beams"]]
    assert scores == sorted(scores, reverse=True)
    assert r["greedy_sequence"] == [0, 0]


# ── grbf16 ───────────────────────────────────────────────────────────

def test_grbf16_relative_error_bounded_by_half_ulp():
    x = lcg_normal(64, seed=23) * 100.0
    r = geron_bf16_range(x)
    # 7 explicit mantissa bits -> half-ulp relative error <= 2^-8.
    assert r["max_rel_error"] <= 2.0**-8 + 1e-12
    assert r["n_overflow"] == 0


def test_grbf16_exact_on_representable_values_and_ties_to_even():
    r = geron_bf16_range([1.0, 1.5, 1.0078125, 256.0, -0.5])
    assert all(r["exact"])
    # 1 + 1/256 is a tie; even mantissa wins -> 1.0.
    assert geron_bf16_range([1.00390625])["bf16"] == [1.0]
    # bf16 keeps fp32's exponent range: 1e38 does not overflow.
    assert geron_bf16_range([1e38])["n_overflow"] == 0


# ── grbgd ────────────────────────────────────────────────────────────

def test_grbgd_first_step_matches_hand_gradient():
    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 2.0, 3.0])
    th0 = np.array([0.5, -0.5])
    eta = 0.05
    g = (2.0 / 3.0) * X.T @ (X @ th0 - y)
    r = geron_batch_gradient_descent(X, y, th0, eta, n_iter=1)
    assert r["theta_path"][1] == pytest.approx(th0 - eta * g)
    assert len(r["loss_history"]) == 2


def test_grbgd_converges_to_normal_equation_solution():
    X = np.column_stack([np.ones(8), lcg(8, seed=29)])
    y = 3.0 - 2.0 * X[:, 1] + 0.01 * lcg_normal(8, seed=31)
    exact = np.linalg.lstsq(X, y, rcond=None)[0]
    r = geron_batch_gradient_descent(X, y, np.zeros(2), 0.4, n_iter=5000)
    assert r["theta"] == pytest.approx(exact, abs=1e-6)
    assert np.all(np.diff(r["loss_history"]) <= 1e-12)   # monotone descent


# ── grblip ───────────────────────────────────────────────────────────

def test_grblip_components_match_independent_formulas():
    I = np.array([[1.0, 0.0], [0.0, 1.0]])
    T = np.array([[1.0, 0.0], [0.0, 1.0]])
    CL = np.zeros((2, 3, 4))                      # uniform over 4 tokens
    tgt = np.array([[0, 1, 2], [3, 0, 1]])
    r = geron_blip_itm_itc(I, T, CL, tgt, tau=1.0)
    # ITC: logits = I2 identity; each row CE = log(1 + e^-1).
    assert r["itc"] == pytest.approx(math.log(1 + math.exp(-1)))
    # ITM: two matched pairs at logit 1, two mismatched at logit 0.
    want_itm = 0.5 * math.log(1 + math.exp(-1)) + 0.5 * math.log(2)
    assert r["itm"] == pytest.approx(want_itm)
    # LM: uniform over 4 -> log 4 per token.
    assert r["lm"] == pytest.approx(math.log(4))
    assert r["lm_perplexity"] == pytest.approx(4.0)
    assert r["loss"] == pytest.approx(r["itc"] + r["itm"] + r["lm"])


def test_grblip_weights_scale_the_terms():
    I = np.array([[1.0, 0.0]])
    r0 = geron_blip_itm_itc(I, I, np.zeros((1, 1, 2)), [[0]], tau=1.0)
    r1 = geron_blip_itm_itc(I, I, np.zeros((1, 1, 2)), [[0]], tau=1.0,
                            lam_lm=3.0)
    assert (r1["loss"] - r0["loss"]) == pytest.approx(2.0 * r0["lm"])


# ── grbn ─────────────────────────────────────────────────────────────

def test_grbn_output_has_zero_mean_unit_variance_before_affine():
    X = lcg_normal(40, seed=37).reshape(10, 4) * 5.0 + 3.0
    r = geron_batch_normalization(X, gamma=1.0, beta=0.0, eps=0.0)
    xh = np.asarray(r["x_hat"])
    assert xh.mean(axis=0) == pytest.approx(np.zeros(4), abs=1e-12)
    assert xh.var(axis=0) == pytest.approx(np.ones(4))
    # Biased variance, as in the paper -- not ddof=1.
    assert r["batch_var"] == pytest.approx(X.var(axis=0, ddof=0))
    assert r["batch_var"] != pytest.approx(X.var(axis=0, ddof=1))


def test_grbn_affine_is_applied_after_normalisation():
    X = lcg_normal(20, seed=43).reshape(5, 4)
    g, b = np.array([2.0, -1.0, 0.5, 3.0]), np.array([1.0, 2.0, 3.0, 4.0])
    r = geron_batch_normalization(X, g, b, eps=1e-5)
    want = g * (X - X.mean(0)) / np.sqrt(X.var(0) + 1e-5) + b
    assert r["Y"] == pytest.approx(want)


def test_grbn_constant_feature_with_zero_eps_raises():
    with pytest.raises(ValueError):
        geron_batch_normalization([[1.0], [1.0]], 1.0, 0.0, eps=0.0)


# ── grbo ─────────────────────────────────────────────────────────────

def test_grbo_fixed_point_satisfies_the_bellman_equation():
    # 2 states, 2 actions, deterministic ring.
    T = np.zeros((2, 2, 2))
    T[0, 0, 0] = 1.0
    T[0, 1, 1] = 1.0
    T[1, 0, 1] = 1.0
    T[1, 1, 0] = 1.0
    R = np.array([[0.0, 1.0], [2.0, -1.0]])
    g = 0.9
    r = geron_bellman_optimality(np.zeros((2, 2)), T, R, g)
    Q = np.asarray(r["Q"])
    V = Q.max(axis=1)
    # Plug the answer back into the equation it claims to solve.
    resid = Q - (R + g * np.einsum("sap,p->sa", T, V))
    assert np.max(np.abs(resid)) < 1e-8
    assert r["converged"]


def test_grbo_geometric_series_closed_form():
    for g in (0.0, 0.5, 0.99):
        r = geron_bellman_optimality([[0.0]], [[[1.0]]], [[[1.0]]], g,
                                     max_iter=20000)
        assert r["Q"][0][0] == pytest.approx(1.0 / (1.0 - g), abs=1e-6)
        assert r["converged"]
    # Hitting the sweep cap is reported, not hidden.
    capped = geron_bellman_optimality([[0.0]], [[[1.0]]], [[[1.0]]], 0.99,
                                      max_iter=10)
    assert not capped["converged"] and capped.warnings


def test_grbo_rejects_non_stochastic_rows():
    with pytest.raises(ValueError):
        geron_bellman_optimality([[0.0]], [[[0.5]]], [[[1.0]]], 0.5)


# ── grbp ─────────────────────────────────────────────────────────────

def test_grbp_gradient_matches_finite_differences():
    rng_a0 = lcg_normal(6, seed=53).reshape(2, 3)
    W1 = lcg_normal(6, seed=59).reshape(3, 2)
    W2 = lcg_normal(2, seed=61).reshape(2, 1)
    y = lcg_normal(2, seed=67).reshape(2, 1)

    def forward(w1, w2):
        a1 = np.tanh(rng_a0 @ w1)
        a2 = np.tanh(a1 @ w2)
        return a1, a2

    a1, a2 = forward(W1, W2)
    r = geron_backpropagation_gradient([rng_a0, a1, a2], [W1, W2], y,
                                       activation="tanh")
    # Central differences on L = 0.5*sum (a2 - y)^2.
    h = 1e-6
    num = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            Wp, Wm = W1.copy(), W1.copy()
            Wp[i, j] += h
            Wm[i, j] -= h
            lp = 0.5 * np.sum((forward(Wp, W2)[1] - y) ** 2)
            lm = 0.5 * np.sum((forward(Wm, W2)[1] - y) ** 2)
            num[i, j] = (lp - lm) / (2 * h)
    assert np.asarray(r["grad_weights"][0]) == pytest.approx(num, abs=1e-6)


def test_grbp_rejects_mismatched_layer_shapes():
    with pytest.raises(ValueError):
        geron_backpropagation_gradient([[[1.0, 2.0]], [[3.0]]],
                                       [[[1.0], [1.0], [1.0]]], [[1.0]])


# ── grbpe ────────────────────────────────────────────────────────────

def test_grbpe_merges_the_most_frequent_pair_first():
    corpus = {"ab": 10, "cb": 3}
    # ('a','b') appears 10 times, ('c','b') 3, ('b','</w>') 13 -> b</w> wins.
    r = geron_bpe_tokenizer_merge(corpus, 1)
    assert r["merges"] == [("b", "</w>")]
    assert r["merge_counts"] == [13]


def test_grbpe_shortens_the_corpus_and_stops_when_no_pair_repeats():
    r = geron_bpe_tokenizer_merge({"low": 5, "lowest": 2}, 20)
    assert r["n_tokens_after"] < r["n_tokens_before"]
    assert r["compression"] > 1.0
    # Every merge must have had count >= 2.
    assert all(c >= 2 for c in r["merge_counts"])
    # Re-joining a split reproduces the word plus the end marker.
    assert "".join(r["splits"]["lowest"]) == "lowest</w>"


def test_grbpe_rejects_empty_corpus():
    with pytest.raises(ValueError):
        geron_bpe_tokenizer_merge([], 3)


# ── grbptt ───────────────────────────────────────────────────────────

def test_grbptt_recurrent_deltas_match_the_backward_recursion():
    G = lcg_normal(6, seed=71).reshape(3, 2)
    H = np.tanh(lcg_normal(6, seed=73).reshape(3, 2))
    X = lcg_normal(9, seed=79).reshape(3, 3)
    Wh = lcg_normal(4, seed=83).reshape(2, 2)
    # Hand-unrolled recursion, last step first. Row-vector convention
    # h_t = tanh(x_t Wx + h_{t-1} Wh) makes dL/dh_{t-1} = Wh @ delta_t.
    d2 = G[2] * (1 - H[2] ** 2)
    d1 = (G[1] + Wh @ d2) * (1 - H[1] ** 2)
    d0 = (G[0] + Wh @ d1) * (1 - H[0] ** 2)
    r = geron_backprop_through_time(G, H, X, W_h=Wh)
    assert r["deltas"] == pytest.approx(np.stack([d0, d1, d2]))
    assert r["grad_Wx"] == pytest.approx(X.T @ np.stack([d0, d1, d2]))


def test_grbptt_gradient_is_summed_over_time_not_averaged():
    G = np.ones((4, 1))
    H = np.zeros((4, 1))
    X = np.ones((4, 1))
    r = geron_backprop_through_time(G, H, X)
    assert r["grad_Wx"][0][0] == pytest.approx(4.0)     # sum, not mean (=1)


def test_grbptt_rejects_hidden_states_outside_tanh_range():
    with pytest.raises(ValueError):
        geron_backprop_through_time([[1.0]], [[2.0]], [[1.0]])


# ── grbrnn ───────────────────────────────────────────────────────────

def test_grbrnn_concat_width_and_content():
    F = lcg_normal(6, seed=89).reshape(3, 2)
    B = lcg_normal(9, seed=97).reshape(3, 3)
    r = geron_bidirectional_rnn(F, B)
    h = np.asarray(r["h"])
    assert h.shape == (3, 5)
    assert h[:, :2] == pytest.approx(F)
    assert h[:, 2:] == pytest.approx(B)


def test_grbrnn_reverse_flag_repairs_the_pairing():
    F = np.array([[1.0], [2.0], [3.0]])
    B_rev = np.array([[30.0], [20.0], [10.0]])     # emitted last-step-first
    r = geron_bidirectional_rnn(F, B_rev, backward_in_reverse_order=True)
    assert r["h"] == [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]


# ── grca ─────────────────────────────────────────────────────────────

def test_grca_matches_explicit_scaled_dot_product():
    Xd = lcg_normal(8, seed=101).reshape(2, 4)
    Xe = lcg_normal(12, seed=103).reshape(3, 4)
    WQ = lcg_normal(8, seed=107).reshape(4, 2)
    WK = lcg_normal(8, seed=109).reshape(4, 2)
    WV = lcg_normal(12, seed=113).reshape(4, 3)
    Q, K, V = Xd @ WQ, Xe @ WK, Xe @ WV
    L = Q @ K.T / math.sqrt(2)
    A = np.exp(L - L.max(1, keepdims=True))
    A = A / A.sum(1, keepdims=True)
    r = geron_cross_attention(Xd, Xe, WQ, WK, WV)
    assert r["attention_weights"] == pytest.approx(A)
    assert r["output"] == pytest.approx(A @ V)
    assert np.asarray(r["attention_weights"]).sum(axis=1) == pytest.approx([1.0, 1.0])


def test_grca_scale_is_one_over_sqrt_dk():
    r = geron_cross_attention([[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]],
                              np.eye(4), np.eye(4), np.eye(4))
    assert r["d_k"] == 4
    assert r["scale"] == pytest.approx(0.5)


# ── grcae ────────────────────────────────────────────────────────────

def test_grcae_loss_equals_explicit_squared_error():
    x = np.arange(1.0, 5.0).reshape(2, 2)
    r = geron_convolutional_autoencoder(x, [[[1.0]]], [[[1.0, 1.0], [1.0, 1.0]]])
    xh = np.asarray(r["x_hat"])
    assert r["loss"] == pytest.approx(float(np.sum((x - xh) ** 2)))
    assert r["code_shape"] == (1, 1)


def test_grcae_shape_mismatch_is_reported_not_cropped():
    with pytest.raises(ValueError):
        geron_convolutional_autoencoder(np.ones((4, 4)), [[[1.0]]], [[[1.0]]])


# ── grcart ───────────────────────────────────────────────────────────

def test_grcart_cost_matches_weighted_gini_by_hand():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 1, 0, 1])
    r = geron_cart_split_cost(X, y, 0, 2.5)
    gl = 1 - (2 / 2) ** 2                      # left = [0,0] -> 0
    gr = 1 - ((1 / 3) ** 2 + (2 / 3) ** 2)     # right = [1,0,1]
    assert r["impurity_left"] == pytest.approx(gl)
    assert r["impurity_right"] == pytest.approx(gr)
    assert r["cost"] == pytest.approx((2 / 5) * gl + (3 / 5) * gr)


def test_grcart_best_threshold_beats_every_other_on_separable_data():
    X = np.array([[v] for v in [1.0, 2.0, 3.0, 4.0]])
    y = np.array([0, 0, 1, 1])
    costs = {t: geron_cart_split_cost(X, y, 0, t)["cost"] for t in (1.5, 2.5, 3.5)}
    assert min(costs, key=costs.get) == 2.5
    assert costs[2.5] == 0.0


def test_grcart_mse_criterion_is_variance_weighted():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 1.0, 5.0, 9.0])
    r = geron_cart_split_cost(X, y, 0, 2.5, criterion="mse")
    right = np.array([5.0, 9.0])
    assert r["cost"] == pytest.approx(0.5 * 0.0 + 0.5 * right.var())


# ── grcfm ────────────────────────────────────────────────────────────

def test_grcfm_counts_and_derived_metrics():
    yt = [0, 0, 1, 1, 2, 2, 2]
    yp = [0, 1, 1, 1, 2, 0, 1]
    r = geron_confusion_matrix(yt, yp)
    cm = np.asarray(r["matrix"])
    for i in range(3):
        for j in range(3):
            assert cm[i, j] == sum(1 for a, b in zip(yt, yp) if a == i and b == j)
    assert cm.sum() == len(yt)
    assert r["accuracy"] == pytest.approx(np.trace(cm) / len(yt))
    # Recall of class 1 = 2/2; precision of class 1 = 2/4.
    assert r["recall"][1] == pytest.approx(1.0)
    assert r["precision"][1] == pytest.approx(0.5)


def test_grcfm_rejects_out_of_range_labels():
    with pytest.raises(ValueError):
        geron_confusion_matrix([0, 1], [0, 5], n_classes=2)


# ── grclp ────────────────────────────────────────────────────────────

def test_grclp_symmetric_and_matches_manual_cross_entropy():
    E = lcg_normal(12, seed=127).reshape(3, 4)
    F = lcg_normal(12, seed=131).reshape(3, 4)
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    Fn = F / np.linalg.norm(F, axis=1, keepdims=True)
    L = En @ Fn.T / 0.5
    def ce(M):
        return float(np.mean([-(M[i, i] - np.log(np.exp(M[i]).sum())) for i in range(3)]))
    r = geron_clip_contrastive_loss(E, F, tau=0.5)
    assert r["loss_i2t"] == pytest.approx(ce(L))
    assert r["loss_t2i"] == pytest.approx(ce(L.T))
    assert r["loss"] == pytest.approx(0.5 * (ce(L) + ce(L.T)))


def test_grclp_degenerate_embeddings_give_chance_loss():
    ones = np.ones((5, 3))
    r = geron_clip_contrastive_loss(ones, ones, tau=1.0)
    assert r["loss"] == pytest.approx(math.log(5))
    assert r["chance_loss"] == pytest.approx(math.log(5))


# ── grcos ────────────────────────────────────────────────────────────

def test_grcos_matches_the_floor_formula_over_a_sweep():
    for in_s in (7, 8, 28, 31):
        for k in (1, 3, 5):
            for p in (0, 1, 2):
                for s in (1, 2, 3):
                    if in_s + 2 * p < k:
                        continue
                    want = (in_s + 2 * p - k) // s + 1
                    got = geron_conv_output_size(in_s, k, p, s)["out_size"][0]
                    assert got == want, (in_s, k, p, s)


def test_grcos_same_padding_preserves_size_for_odd_kernels():
    for k in (1, 3, 5, 7):
        p = geron_conv_output_size(32, k)["same_padding"][0]
        assert geron_conv_output_size(32, k, p, 1)["out_size"] == [32]


def test_grcos_rejects_oversized_kernel():
    with pytest.raises(ValueError):
        geron_conv_output_size(3, 5)


# ── grctr ────────────────────────────────────────────────────────────

def test_grctr_matches_manual_softmax_cross_entropy():
    a = lcg_normal(3, seed=137).reshape(1, 3)
    p = lcg_normal(3, seed=139).reshape(1, 3)
    n = lcg_normal(9, seed=149).reshape(3, 3)
    an = a / np.linalg.norm(a)
    pn = p / np.linalg.norm(p)
    nn = n / np.linalg.norm(n, axis=1, keepdims=True)
    tau = 0.2
    sp = float((an @ pn.T).item()) / tau
    sn = (an @ nn.T).ravel() / tau
    want = -(sp - np.log(np.exp(sp) + np.exp(sn).sum()))
    assert geron_contrastive_infonce(a, p, n, tau=tau)["loss"] == pytest.approx(want)


def test_grctr_loss_decreases_as_positive_gets_closer():
    a = np.array([[1.0, 0.0]])
    neg = np.array([[0.0, 1.0]])
    far = geron_contrastive_infonce(a, [[0.2, 1.0]], neg, tau=0.5)["loss"]
    near = geron_contrastive_infonce(a, [[1.0, 0.05]], neg, tau=0.5)["loss"]
    assert near < far
    assert geron_contrastive_infonce(a, a, neg, tau=0.5)["accuracy"] == 1.0


# ── grcvf ────────────────────────────────────────────────────────────

def test_grcvf_matches_explicit_windowed_sum():
    X = lcg_normal(25, seed=151).reshape(5, 5)
    W = lcg_normal(9, seed=157).reshape(3, 3)
    r = geron_conv2d_forward(X, W, b=0.5, stride=2, padding=1)
    Xp = np.pad(X, 1)
    Y = np.asarray(r["Y"])
    assert Y.shape == (3, 3)
    for i in range(3):
        for j in range(3):
            assert Y[i, j] == pytest.approx(
                float(np.sum(Xp[2 * i:2 * i + 3, 2 * j:2 * j + 3] * W)) + 0.5)


def test_grcvf_output_shape_agrees_with_grcos():
    for k, p, s in ((3, 1, 1), (2, 0, 2), (5, 2, 3)):
        X = np.ones((9, 9))
        W = np.ones((k, k))
        got = geron_conv2d_forward(X, W, stride=s, padding=p)["out_shape"]
        want = tuple(geron_conv_output_size([9, 9], k, p, s)["out_size"])
        assert got == want


def test_grcvf_is_cross_correlation_not_flipped_convolution():
    X = [[1.0, 0.0], [0.0, 0.0]]
    W = [[1.0, 2.0], [3.0, 4.0]]
    # Cross-correlation lines W[0,0] up with X[0,0] -> 1*1 = 1.
    assert geron_conv2d_forward(X, W)["Y"] == [[1.0]]


# ── grcvs ────────────────────────────────────────────────────────────

def test_grcvs_averages_the_per_fold_scores_and_partitions_the_data():
    X = np.column_stack([np.ones(9), lcg(9, seed=163)])
    y = 1.0 + 2.0 * X[:, 1]
    r = geron_cross_validation_score(X, y, K=3)
    assert sum(r["fold_sizes"]) == 9
    assert r["cv_score"] == pytest.approx(float(np.mean(r["fold_scores"])))
    assert r["cv_score"] == pytest.approx(1.0)          # noiseless -> R^2 = 1


def test_grcvs_uses_a_caller_supplied_model():
    calls = []

    def fit(Xtr, ytr):
        calls.append(len(ytr))
        return float(np.mean(ytr))

    def predict(model, Xte):
        return np.full(Xte.shape[0], model)

    X = np.arange(8.0).reshape(8, 1)
    y = np.array([1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0])
    r = geron_cross_validation_score(X, y, K=4, fit=fit, predict=predict,
                                     score=lambda a, b: -float(np.mean((a - b) ** 2)))
    assert calls == [6, 6, 6, 6]                        # K-1 folds of training data
    assert r["K"] == 4 and len(r["fold_scores"]) == 4


def test_grcvs_rejects_too_many_folds():
    with pytest.raises(ValueError):
        geron_cross_validation_score(np.ones((3, 1)), np.arange(3.0), K=5)


# ── grdae ────────────────────────────────────────────────────────────

def test_grdae_scores_against_clean_input_not_corrupted():
    x = lcg_normal(8, seed=167).reshape(2, 4)
    noise = lcg_normal(8, seed=173).reshape(2, 4)
    # Reconstructing the CORRUPTED input must cost the noise energy.
    r = geron_denoising_autoencoder(x, noise, x + noise)
    assert r["loss"] == pytest.approx(float(np.mean(np.sum(noise**2, axis=1))))
    assert r["denoising_gain"] == pytest.approx(1.0)
    # Reconstructing the clean input is free.
    assert geron_denoising_autoencoder(x, noise, x)["loss"] == pytest.approx(0.0)


def test_grdae_dropout_mask_multiplies():
    r = geron_denoising_autoencoder([[1.0, 2.0]], [[1.0, 0.0]], [[1.0, 2.0]],
                                    corruption="dropout")
    assert r["x_tilde"] == [[1.0, 0.0]]
    with pytest.raises(ValueError):
        geron_denoising_autoencoder([[1.0]], [[0.5]], [[1.0]], corruption="dropout")


# ── grdal ────────────────────────────────────────────────────────────

def test_grdal_likelihood_is_the_product_over_positions():
    logits = {0: np.array([0.0, math.log(3.0)]), 1: np.array([math.log(2.0), 0.0])}

    def fn(ctx):
        return logits[len(ctx) % 2]

    r = geron_dalle_autoregressive_token([9], [1, 0], fn)
    # step 0: ctx len 1 -> logits[1], token 1 -> p = 1/(2+1) = 1/3
    # step 1: ctx len 2 -> logits[0], token 0 -> p = 1/(1+3) = 1/4
    assert r["token_logprobs"] == pytest.approx([math.log(1 / 3), math.log(1 / 4)])
    assert r["log_likelihood"] == pytest.approx(math.log(1 / 12))
    assert r["perplexity"] == pytest.approx(12 ** 0.5)


def test_grdal_enforces_the_callable_contract():
    with pytest.raises(ValueError):
        geron_dalle_autoregressive_token([0], [0], lambda ctx: np.array([np.inf, 0.0]))
    with pytest.raises(ValueError):
        geron_dalle_autoregressive_token([0], [5], lambda ctx: np.zeros(2))
    with pytest.raises(ValueError):
        geron_dalle_autoregressive_token([0], [0], "not callable")


# ── grdbs ────────────────────────────────────────────────────────────

def test_grdbs_counts_include_self_and_match_brute_force():
    X = lcg_normal(20, seed=179).reshape(10, 2)
    eps = 1.2
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(2))
    want = (D <= eps).sum(axis=1)
    r = geron_dbscan_core_point(X, eps, 3)
    assert r["neighbor_counts"] == want.tolist()
    assert r["is_core"] == (want >= 3).tolist()
    # min_samples=1 makes every point core (self counts).
    assert all(geron_dbscan_core_point(X, eps, 1)["is_core"])


def test_grdbs_partitions_into_core_border_noise():
    X = [[0.0], [0.4], [0.8], [10.0]]
    r = geron_dbscan_core_point(X, 0.5, 2)
    core = np.asarray(r["is_core"])
    border = np.asarray(r["is_border"])
    noise = np.asarray(r["is_noise"])
    assert np.all(core.astype(int) + border.astype(int) + noise.astype(int) == 1)
    assert noise[3]


# ── grdcgan ──────────────────────────────────────────────────────────

def test_grdcgan_upsamples_by_stride_and_bounds_output_in_tanh_range():
    W0 = lcg_normal(16, seed=181).reshape(2, 8)      # z(2) -> 8 = 4x2 seed
    K1 = lcg_normal(9, seed=191).reshape(3, 3)
    K2 = lcg_normal(16, seed=193).reshape(4, 4)
    r = geron_dcgan_generator(lcg_normal(2, seed=197), [W0, K1, K2],
                              seed_shape=(4, 2))
    img = np.asarray(r["image"])
    # transposed conv: (n-1)*s + k, twice.
    h = (4 - 1) * 2 + 3
    h = (h - 1) * 2 + 4
    w = (2 - 1) * 2 + 3
    w = (w - 1) * 2 + 4
    assert r["image_shape"] == (h, w)
    assert np.all(np.abs(img) < 1.0)                 # tanh output


def test_grdcgan_transposed_conv_scatters_and_sums():
    # 2x2 seed of ones, 2x2 kernel of ones, stride 1 -> overlaps add up.
    W0 = [[1.0, 1.0, 1.0, 1.0]]
    K = [[1.0, 1.0], [1.0, 1.0]]
    r = geron_dcgan_generator([1.0], [W0, K], seed_shape=(2, 2), stride=1)
    # pre-tanh output is [[1,2,1],[2,4,2],[1,2,1]]
    assert np.asarray(r["image"]) == pytest.approx(
        np.tanh(np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])))


def test_grdcgan_rejects_non_square_projection_without_seed_shape():
    with pytest.raises(ValueError):
        geron_dcgan_generator([1.0], [[[1.0, 1.0, 1.0]], [[1.0]]])


# ── grddim ───────────────────────────────────────────────────────────

def test_grddim_step_is_the_two_stage_formula():
    ab = [1.0, 0.9, 0.7, 0.4]
    x_t = lcg_normal(4, seed=199)
    eps = lcg_normal(4, seed=211)
    r = geron_ddim_sampling_step(x_t, 3, 1, eps, ab)
    x0 = (x_t - math.sqrt(1 - 0.4) * eps) / math.sqrt(0.4)
    want = math.sqrt(0.9) * x0 + math.sqrt(1 - 0.9) * eps
    assert r["x0_pred"] == pytest.approx(x0)
    assert r["x_prev"] == pytest.approx(want)


def test_grddim_is_an_identity_when_the_prediction_is_consistent():
    # If x_t was built as sqrt(ab_t)*x0 + sqrt(1-ab_t)*eps, the step must
    # return exactly sqrt(ab_prev)*x0 + sqrt(1-ab_prev)*eps.
    ab = [1.0, 0.8, 0.5]
    x0 = np.array([1.0, -2.0, 0.5])
    eps = np.array([0.3, 0.1, -0.7])
    x_t = math.sqrt(0.5) * x0 + math.sqrt(0.5) * eps
    r = geron_ddim_sampling_step(x_t, 2, 1, eps, ab)
    assert r["x0_pred"] == pytest.approx(x0)
    assert r["x_prev"] == pytest.approx(math.sqrt(0.8) * x0 + math.sqrt(0.2) * eps)


def test_grddim_rejects_forward_steps_and_bad_schedules():
    with pytest.raises(ValueError):
        geron_ddim_sampling_step([1.0], 1, 2, [0.0], [1.0, 0.8, 0.5])
    with pytest.raises(ValueError):
        geron_ddim_sampling_step([1.0], 2, 1, [0.0], [0.5, 0.8, 1.0])


# ── grddqn ───────────────────────────────────────────────────────────

def test_grddqn_evaluates_the_online_argmax_with_the_target_net():
    Qo = lcg_normal(12, seed=223).reshape(3, 4)
    Qt = lcg_normal(12, seed=227).reshape(3, 4)
    sn = [0, 2, 1]
    rew = [1.0, -0.5, 0.0]
    g = 0.95
    want = [rew[k] + g * Qt[sn[k], int(np.argmax(Qo[sn[k]]))] for k in range(3)]
    r = geron_double_dqn_target(Qo, Qt, sn, rew, g)
    assert r["target"] == pytest.approx(want)
    # The vanilla target is never smaller -- that is the overestimation.
    assert np.all(np.asarray(r["overestimation_gap"]) >= -1e-12)


def test_grddqn_matches_vanilla_when_both_nets_agree():
    Q = lcg_normal(8, seed=229).reshape(2, 4)
    r = geron_double_dqn_target(Q, Q, [0, 1], [0.0, 0.0], 0.9)
    assert r["target"] == pytest.approx(r["vanilla_target"])


# ── grdeit ───────────────────────────────────────────────────────────

def test_grdeit_is_the_weighted_sum_of_two_cross_entropies():
    Lc = lcg_normal(6, seed=233).reshape(2, 3)
    Ld = lcg_normal(6, seed=239).reshape(2, 3)
    y = np.array([0, 2])
    teach = lcg_normal(6, seed=241).reshape(2, 3)
    tl = teach.argmax(axis=1)

    def ce(Z, lab):
        return float(np.mean([-(Z[i, lab[i]] - np.log(np.exp(Z[i]).sum()))
                              for i in range(2)]))

    r = geron_deit_distillation_loss(Lc, Ld, y, teach, alpha=0.25)
    assert r["loss_cls"] == pytest.approx(ce(Lc, y))
    assert r["loss_dist"] == pytest.approx(ce(Ld, tl))
    assert r["loss"] == pytest.approx(0.75 * ce(Lc, y) + 0.25 * ce(Ld, tl))
    assert r["teacher_labels"] == tl.tolist()


def test_grdeit_alpha_endpoints_select_one_head():
    Lc = np.array([[5.0, 0.0]])
    Ld = np.array([[0.0, 5.0]])
    a0 = geron_deit_distillation_loss(Lc, Ld, [0], [[1.0, 0.0]], alpha=0.0)
    a1 = geron_deit_distillation_loss(Lc, Ld, [0], [[1.0, 0.0]], alpha=1.0)
    assert a0["loss"] == pytest.approx(a0["loss_cls"])
    assert a1["loss"] == pytest.approx(a1["loss_dist"])


# ── grdetr ───────────────────────────────────────────────────────────

def _brute_force_assignment(C):
    """Exhaustive minimum-cost matching of rows onto distinct columns."""
    import itertools
    n, m = C.shape
    best, best_cost = None, math.inf
    for cols in itertools.permutations(range(m), n):
        c = sum(C[i, cols[i]] for i in range(n))
        if c < best_cost:
            best, best_cost = cols, c
    return best, best_cost


def test_grdetr_hungarian_agrees_with_brute_force():
    # Random cost via the module's own matcher, checked exhaustively.
    from morie.fn.grdetr import _linear_sum_assignment
    for seed in (251, 257, 263):
        C = np.round(lcg_normal(20, seed=seed).reshape(4, 5), 4)
        rows, cols = _linear_sum_assignment(C)
        got = float(C[rows, cols].sum())
        _, want = _brute_force_assignment(C)
        assert got == pytest.approx(want)
        assert sorted(rows.tolist()) == [0, 1, 2, 3]
        assert len(set(cols.tolist())) == 4


def test_grdetr_matching_is_one_to_one_and_picks_the_overlapping_box():
    pb = [[0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 6.0, 6.0], [0.1, 0.1, 2.1, 2.1]]
    pc = [[3.0, 0.0], [0.0, 3.0], [0.0, 3.0]]
    gb = [[0.0, 0.0, 2.0, 2.0]]
    r = geron_detr_hungarian_matching(pb, pc, gb, [0])
    assert r["matching"] == [(0, 0)]
    assert r["loss_bbox"] == 0.0
    assert r["matched_giou"] == pytest.approx([1.0])
    assert r["unmatched_predictions"] == [1, 2]


def test_grdetr_giou_is_negative_for_disjoint_boxes():
    pb = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
    pc = [[1.0, 0.0], [1.0, 0.0]]
    gb = [[10.0, 10.0, 11.0, 11.0]]
    r = geron_detr_hungarian_matching(pb, pc, gb, [0])
    # union = 2, enclosing = 11*11 = 121, iou = 0 -> giou = -(121-2)/121
    assert r["matched_giou"][0] == pytest.approx(-(121 - 2) / 121)
    assert r["loss_giou"] == pytest.approx(2.0 * (1 - r["matched_giou"][0]))


def test_grdetr_no_object_term_penalises_spare_queries():
    pb = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
    pc = [[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]]
    gb = [[0.0, 0.0, 1.0, 1.0]]
    with_eos = geron_detr_hungarian_matching(pb, pc, gb, [0], no_object_class=2)
    without = geron_detr_hungarian_matching(pb, pc, gb, [0])
    assert with_eos["loss_no_object"] > 0.0
    assert with_eos["loss"] > without["loss"]


def test_grdetr_rejects_fewer_queries_than_objects():
    with pytest.raises(ValueError):
        geron_detr_hungarian_matching([[0.0, 0.0, 1.0, 1.0]], [[1.0, 0.0]],
                                      [[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]],
                                      [0, 1])


# ── a mean-of-inputs stub must fail every one of these ───────────────

def test_placeholder_bodies_would_be_caught():
    """Guard: the old placeholder returned float(np.mean(inputs))."""
    x = [[3.0, 5.0], [7.0, 9.0]]
    stub = float(np.mean(x))
    assert geron_batch_normalization(x, 1.0, 0.0)["estimate"] != pytest.approx(stub)
    assert geron_conv2d_forward(x, [[1.0, 1.0]])["estimate"] != pytest.approx(stub)
    assert float(geron_1cycle_schedule(0.1, 0.5, 1, 5)) != pytest.approx(
        float(np.mean([0.1, 0.5, 1, 5])))
    assert geron_adam_update([1.0, 2.0], [0.1, 0.2], [0.0, 0.0], [0.0, 0.0],
                             1, 0.01)["estimate"] != pytest.approx(1.5)
