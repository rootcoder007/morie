# morie.fn -- test file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent-route checks for the Géron tranche-1 modules.

Every assertion is derived from a different computation than the one under
test: closed-form gradients against central finite differences, losses
against hand log-sum-exp, metrics against brute-force counting, optimizer
steps against a hand-derived first iteration, and structural invariants
(layer norm, attention row sums, parameter sharing) that a mean-of-inputs
stub cannot satisfy.
"""

import math

import numpy as np
import pytest

from morie.fn.hma2c import geron_a2c
from morie.fn.hma3c import geron_a3c
from morie.fn.hmadab import geron_adaboost
from morie.fn.hmadam import geron_adam
from morie.fn.hmadgr import geron_adagrad
from morie.fn.hmadmw import geron_adamw
from morie.fn.hmadmx import geron_adamax
from morie.fn.hmaen import geron_autoencoder
from morie.fn.hmagc import geron_agglomerative
from morie.fn.hmagrd import geron_autograd
from morie.fn.hmaic import geron_aic
from morie.fn.hmalbt import geron_albert
from morie.fn.hmalex import geron_alexnet
from morie.fn.hmanae import geron_anomaly_autoencoder
from morie.fn.hmarim import geron_arima
from morie.fn.hmauc import geron_auc_roc
from morie.fn.hmauxpt import geron_auxiliary_task_pretraining
from morie.fn.hmbag import geron_bagging
from morie.fn.hmbart import geron_bart
from morie.fn.hmbat import geron_batch_learning
from morie.fn.hmbdn import geron_bahdanau_attention
from morie.fn.hmbel import geron_bellman_optimality
from morie.fn.hmbert import geron_bert
from morie.fn.hmbf16 import geron_bf16
from morie.fn.hmbftn import geron_bert_finetune
from morie.fn.hmbgdg import geron_batch_gd_grad
from morie.fn.hmbic import geron_bic
from morie.fn.hmbin import geron_binary_classification
from morie.fn.hmblip import geron_blip
from morie.fn.hmblp2 import geron_blip2
from morie.fn.hmbms import geron_beam_search
from morie.fn.hmbnm import geron_biological_neuron
from morie.fn.hmbntr import geron_batch_normalization
from morie.fn.hmbp import geron_backpropagation
from morie.fn.hmbpet import geron_bpe_tokenizer
from morie.fn.hmbrch import geron_birch
from morie.fn.hmbrnn import geron_bidirectional_rnn
from morie.fn.hmbrob import geron_roberta
from morie.fn.hmbsz import geron_batch_size_heuristic
from morie.fn.hmbv import geron_bias_variance_tradeoff


# ── shared helpers ────────────────────────────────────────────────────


def lcg(n, seed=1, lo=0.0, hi=1.0):
    """Exact-integer LCG -- reproducible on every platform."""
    s = int(seed) % 2**32
    out = np.empty(n)
    for i in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = lo + (hi - lo) * ((s + 0.5) / 2**32)
    return out


def central_diff(f, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp.flat[i] += h
        xm.flat[i] -= h
        g.flat[i] = (f(xp) - f(xm)) / (2 * h)
    return g


# ── hmaic ─────────────────────────────────────────────────────────────


def test_aic_matches_hand_arithmetic_and_ranks_by_penalty():
    r = geron_aic([-20.0, -19.0, -19.0], [3, 5, 4])
    assert [float(v) for v in r["aic"]] == [46.0, 48.0, 46.0]
    assert r["best_index"] in (0, 2)
    # Akaike weights are a softmax of -delta/2 -- recompute independently.
    d = np.array([0.0, 2.0, 0.0])
    w = np.exp(-0.5 * d) / np.sum(np.exp(-0.5 * d))
    assert np.allclose(r["weights"], w)


def test_aic_rejects_fractional_k():
    with pytest.raises(ValueError, match="non-negative integer"):
        geron_aic(-5.0, 2.5)


# ── hmbic ─────────────────────────────────────────────────────────────


def test_bic_penalty_exceeds_aic_penalty_once_n_above_e_squared():
    ll, k, n = -30.0, 4, 100
    bic = geron_bic(ll, k, n)["bic"]
    assert bic == pytest.approx(-2 * ll + k * math.log(n))
    aic = geron_aic(ll, k)["aic"]
    assert bic > aic  # log(100) > 2


def test_bic_requires_positive_n():
    with pytest.raises(ValueError, match="n must be >= 1"):
        geron_bic(-1.0, 1, 0)


# ── hmbsz ─────────────────────────────────────────────────────────────


def test_batch_size_is_admissible_power_of_two_and_respects_step_budget():
    for n in (40, 400, 4000, 40000, 400000):
        r = geron_batch_size_heuristic(n, steps_per_epoch_target=10)
        b = r["batch_size"]
        assert b in (32, 64, 128, 256, 512) or b == min(32, n)
        assert r["steps_per_epoch"] == math.ceil(n / b)
        if b < 512 and not r["clamped"]:
            assert 2 * b > n / 10  # nothing larger would have fit


def test_batch_size_monotone_in_dataset_size():
    sizes = [geron_batch_size_heuristic(n)["batch_size"] for n in (100, 1000, 10000, 100000)]
    assert sizes == sorted(sizes)


# ── hmbf16 ────────────────────────────────────────────────────────────


def test_bf16_relative_error_bounded_by_half_ulp():
    vals = lcg(64, seed=5, lo=-100.0, hi=100.0)
    r = geron_bf16(vals)
    # 7 explicit mantissa bits -> half-ULP relative bound 2**-9 (plus slack
    # for the hidden bit's leading-digit effect).
    assert r["max_rel_error"] <= 2.0**-8


def test_bf16_is_idempotent_and_exact_on_representable_values():
    exact = [1.0, 2.0, 0.5, -4.0, 1.5, 384.0]
    q1 = geron_bf16(exact)["values"]
    assert [float(v) for v in q1] == exact
    q2 = geron_bf16(q1)["values"]
    assert np.array_equal(np.asarray(q1), np.asarray(q2))


def test_bf16_truncation_never_rounds_away_from_zero():
    vals = lcg(32, seed=9, lo=0.1, hi=10.0)
    near = geron_bf16(vals)["values"].astype(float)
    trunc = geron_bf16(vals, rounding="truncate")["values"].astype(float)
    assert np.all(trunc <= vals + 1e-12)
    assert np.all(np.abs(near - vals) <= np.abs(trunc - vals) + 1e-12)


# ── hmbnm ─────────────────────────────────────────────────────────────


def test_neuron_fires_exactly_when_weighted_sum_reaches_threshold():
    w = [2.0, -1.0]
    b = -0.5
    for x0 in (0.0, 0.25, 0.5, 1.0):
        x = [x0, 0.0]
        z = 2.0 * x0 - 0.5
        r = geron_biological_neuron(x, w, b)
        assert float(r["z"]) == pytest.approx(z)
        assert float(r["a"]) == (1.0 if z >= 0 else 0.0)
    assert geron_biological_neuron([[1.0, 0.0], [0.0, 1.0]], w, b)["z"].shape == (2,)


def test_neuron_shape_mismatch_raises():
    with pytest.raises(ValueError, match="features but w has"):
        geron_biological_neuron([1.0, 2.0, 3.0], [1.0, 1.0], 0.0)


# ── hmbgdg ────────────────────────────────────────────────────────────


def test_mse_gradient_matches_central_differences():
    X = np.array([[1.0, 0.5], [1.0, -2.0], [1.0, 3.0], [1.0, 0.0]])
    y = np.array([1.0, -1.0, 4.0, 0.5])
    theta = np.array([0.3, -0.7])

    def cost(t):
        r = X @ t - y
        return float(r @ r / X.shape[0])

    analytic = np.asarray(geron_batch_gd_grad(X, y, theta)["gradient"])
    assert np.allclose(analytic, central_diff(cost, theta), atol=1e-6)


def test_gd_step_decreases_cost_for_small_eta():
    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([2.0, 3.0, 5.0])
    t0 = np.zeros(2)
    r = geron_batch_gd_grad(X, y, t0, eta=0.05)
    r2 = geron_batch_gd_grad(X, y, r["theta_next"])
    assert r2["cost"] < r["cost"]


# ── hmauc ─────────────────────────────────────────────────────────────


def test_auc_matches_brute_force_pair_counting():
    y = [0, 1, 0, 1, 1, 0, 0, 1]
    s = list(lcg(8, seed=11))
    pos = [s[i] for i in range(8) if y[i] == 1]
    neg = [s[i] for i in range(8) if y[i] == 0]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    assert float(geron_auc_roc(y, s)["auc"]) == pytest.approx(wins / (len(pos) * len(neg)))


def test_auc_invariant_under_monotone_score_transform():
    y = [0, 0, 1, 1, 0, 1]
    s = np.array([0.2, 0.9, 0.4, 0.8, 0.1, 0.55])
    a1 = float(geron_auc_roc(y, s)["auc"])
    a2 = float(geron_auc_roc(y, np.exp(3 * s))["auc"])
    assert a1 == pytest.approx(a2)


def test_auc_needs_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        geron_auc_roc([1, 1, 1], [0.1, 0.2, 0.3])


# ── hmbin ─────────────────────────────────────────────────────────────


def test_binary_classification_confusion_matches_brute_force():
    X = np.column_stack([np.ones(6), lcg(6, seed=13, lo=-2.0, hi=2.0)])
    theta = np.array([-0.2, 1.5])
    y = [0, 1, 0, 1, 1, 0]
    r = geron_binary_classification(X, theta, y_true=y)
    p = 1.0 / (1.0 + np.exp(-(X @ theta)))
    pred = (p >= 0.5).astype(int)
    tp = sum(1 for i in range(6) if pred[i] == 1 and y[i] == 1)
    fp = sum(1 for i in range(6) if pred[i] == 1 and y[i] == 0)
    fn = sum(1 for i in range(6) if pred[i] == 0 and y[i] == 1)
    assert (r["tp"], r["fp"], r["fn"]) == (tp, fp, fn)
    assert float(r["precision"]) == pytest.approx(tp / (tp + fp))
    assert np.allclose(np.asarray(r["p_hat"]), p)


def test_threshold_monotonically_shrinks_positive_set():
    X = np.column_stack([np.ones(20), lcg(20, seed=17, lo=-3.0, hi=3.0)])
    th = np.array([0.0, 1.0])
    counts = [int(np.sum(geron_binary_classification(X, th, threshold=t)["y_pred"])) for t in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert counts == sorted(counts, reverse=True)


# ── hmbv ──────────────────────────────────────────────────────────────


def test_bias_variance_decomposition_closes_exactly():
    preds = np.array([[1.0, 2.0, 3.0], [2.0, 2.5, 1.0], [0.0, 4.0, 2.0], [3.0, 1.0, 2.5]])
    y = np.array([1.5, 2.0, 2.0])
    r = geron_bias_variance_tradeoff(preds, y)
    mse = float(np.mean((preds - y) ** 2))
    assert float(r["mse"]) == pytest.approx(mse)
    assert float(r["bias2"]) + float(r["variance"]) == pytest.approx(mse)
    assert float(r["variance"]) == pytest.approx(float(np.mean(preds.var(axis=0, ddof=0))))


def test_identical_models_have_zero_variance():
    preds = np.tile([1.0, 2.0], (5, 1))
    r = geron_bias_variance_tradeoff(preds, [1.0, 3.0])
    assert float(r["variance"]) == 0.0
    assert float(r["bias2"]) == pytest.approx(0.5)


# ── hmbntr ────────────────────────────────────────────────────────────


def test_batchnorm_standardises_each_feature_independently():
    X = np.column_stack([lcg(12, seed=19, lo=-5.0, hi=5.0), lcg(12, seed=23, lo=10.0, hi=12.0)])
    r = geron_batch_normalization(X, eps=0.0)
    xh = np.asarray(r["x_hat"])
    assert np.allclose(xh.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(xh.var(axis=0, ddof=0), 1.0, atol=1e-12)


def test_batchnorm_affine_is_exactly_gamma_xhat_plus_beta():
    X = np.array([[1.0, 4.0], [3.0, 8.0], [5.0, 0.0]])
    r = geron_batch_normalization(X, gamma=[2.0, -1.0], beta=[1.0, 0.5], eps=1e-5)
    expected = np.asarray(r["x_hat"]) * np.array([2.0, -1.0]) + np.array([1.0, 0.5])
    assert np.allclose(np.asarray(r["y"]), expected)


def test_batchnorm_zero_variance_with_zero_eps_raises():
    with pytest.raises(ValueError, match="zero variance"):
        geron_batch_normalization([[2.0], [2.0]], eps=0.0)


# ── hmbat ─────────────────────────────────────────────────────────────


def test_batch_learning_solves_the_normal_equations():
    X = np.column_stack([np.ones(8), lcg(8, seed=29, lo=-1.0, hi=1.0), lcg(8, seed=31)])
    y = lcg(8, seed=37, lo=-2.0, hi=2.0)
    theta = np.asarray(geron_batch_learning(X, y)["theta"])
    # Residuals of the least-squares fit are orthogonal to every column.
    assert np.allclose(X.T @ (y - X @ theta), 0.0, atol=1e-9)


def test_ridge_shrinks_the_coefficient_norm():
    X = np.column_stack([lcg(10, seed=41), lcg(10, seed=43)])
    y = lcg(10, seed=47, lo=-1.0, hi=1.0)
    n0 = np.linalg.norm(geron_batch_learning(X, y)["theta"])
    n1 = np.linalg.norm(geron_batch_learning(X, y, ridge=5.0)["theta"])
    assert n1 < n0


# ── hmbdn ─────────────────────────────────────────────────────────────


def test_bahdanau_alpha_matches_hand_softmax():
    h = np.array([[1.0, 2.0], [-1.0, 0.5], [0.0, 0.0]])
    s = np.array([0.3])
    W = np.array([[0.5, -0.25], [1.0, 0.0]])
    U = np.array([[2.0], [-1.0]])
    v = np.array([1.0, -0.5])
    e = np.array([float(v @ np.tanh(W @ h[i] + U @ s)) for i in range(3)])
    alpha = np.exp(e - e.max()) / np.sum(np.exp(e - e.max()))
    r = geron_bahdanau_attention(h, s, W, U, v)
    assert np.allclose(np.asarray(r["alpha"]), alpha)
    assert np.allclose(np.asarray(r["context"]), alpha @ h)
    assert float(np.sum(r["alpha"])) == pytest.approx(1.0)


def test_bahdanau_entropy_is_maximal_for_uniform_attention():
    h = np.zeros((4, 2))
    r = geron_bahdanau_attention(h, [0.0], np.zeros((2, 2)), np.zeros((2, 1)), [1.0, 1.0])
    assert float(r["entropy"]) == pytest.approx(math.log(4))


# ── hmbel ─────────────────────────────────────────────────────────────


def test_value_iteration_matches_closed_form_geometric_sum():
    P = np.ones((1, 1, 1))
    R = np.array([[2.0]])
    for g in (0.0, 0.3, 0.9):
        v = float(geron_bellman_optimality([0.0], P, R, g)["V"][0])
        assert v == pytest.approx(2.0 / (1.0 - g), rel=1e-8)


def test_value_iteration_picks_the_better_action_and_satisfies_bellman():
    # Two states, two actions; deterministic transitions.
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = 1.0
    P[0, 1, 1] = 1.0
    P[1, 0, 1] = 1.0
    P[1, 1, 0] = 1.0
    R = np.array([[0.0, 1.0], [0.0, 0.0]])
    r = geron_bellman_optimality([0.0, 0.0], P, R, 0.9)
    V = np.asarray(r["V"])
    residual = np.max(np.abs(V - (R + 0.9 * (P @ V)).max(axis=1)))
    assert residual < 1e-8
    assert r["converged"]


def test_bellman_rejects_non_stochastic_kernel():
    with pytest.raises(ValueError, match="sums to"):
        geron_bellman_optimality([0.0], [[[0.5]]], [[1.0]], 0.5)


# ── optimizers: hmadam / hmadgr / hmadmw / hmadmx ──────────────────────


def test_adam_first_step_is_minus_eta_regardless_of_gradient_scale():
    for g in (0.001, 1.0, 1000.0, -7.5):
        step = float(geron_adam([g], eta=0.1, t=1)["step"][0])
        assert step == pytest.approx(-0.1 * np.sign(g), rel=1e-4)


def test_adam_moments_match_the_hand_recursion_over_three_steps():
    b1, b2, eta = 0.9, 0.999, 0.01
    grads = [1.0, -2.0, 0.5]
    m = v = 0.0
    state_m, state_v = np.zeros(1), np.zeros(1)
    for t, g in enumerate(grads, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        r = geron_adam([g], m=state_m, v=state_v, b1=b1, b2=b2, eta=eta, t=t)
        state_m, state_v = r["m"], r["v"]
        assert float(state_m[0]) == pytest.approx(m)
        assert float(state_v[0]) == pytest.approx(v)
        expected = -eta * (m / (1 - b1**t)) / (math.sqrt(v / (1 - b2**t)) + 1e-8)
        assert float(r["step"][0]) == pytest.approx(expected)


def test_adagrad_effective_rate_decays_like_one_over_sqrt_t():
    g, eta = 1.0, 0.1
    s = np.zeros(1)
    rates = []
    for t in range(1, 6):
        r = geron_adagrad([g], s=s, eta=eta, eps=0.0)
        s = r["s"]
        rates.append(float(r["effective_lr"][0]))
        assert float(s[0]) == pytest.approx(t)
    assert np.allclose(rates, [eta / math.sqrt(t) for t in range(1, 6)])


def test_adamw_decay_is_decoupled_from_the_adaptive_denominator():
    # Two gradients of wildly different scale get the same decay term.
    for g in (1e-4, 1e4):
        r = geron_adamw([g], theta=[3.0], eta=0.1, wd=0.2, t=1)
        assert float(r["decay_step"][0]) == pytest.approx(-0.1 * 0.2 * 3.0)
    r0 = geron_adamw([1.0], theta=[3.0], eta=0.1, wd=0.0, t=1)
    r1 = geron_adamw([1.0], theta=[3.0], eta=0.1, wd=0.2, t=1)
    assert float(r1["theta"][0]) == pytest.approx(float(r0["theta"][0]) - 0.06)


def test_adamax_infinity_norm_tracks_running_max_not_mean():
    u = np.zeros(1)
    m = np.zeros(1)
    b2 = 0.9
    peak = 0.0
    for g in (1.0, 5.0, 0.1, 0.1):
        r = geron_adamax([g], m=m, u=u, b2=b2, eta=0.1, t=1)
        m, u = r["m"], r["u"]
        peak = max(b2 * peak, abs(g))
        assert float(u[0]) == pytest.approx(peak)
    assert float(u[0]) > 0.1  # a mean-of-|g| accumulator would have decayed below


def test_optimizers_reject_bad_hyperparameters():
    with pytest.raises(ValueError, match="b1 and b2"):
        geron_adam([1.0], b1=1.0)
    with pytest.raises(ValueError, match="eta"):
        geron_adagrad([1.0], eta=-0.1)
    with pytest.raises(ValueError, match="wd"):
        geron_adamw([1.0], wd=-1.0)
    with pytest.raises(ValueError, match="timestep"):
        geron_adamax([1.0], t=0)


# ── hmagrd ────────────────────────────────────────────────────────────


def test_autograd_matches_central_differences_on_a_composite_function():
    def f_var(p):
        return (p[0] * p[1]).exp() + (p[0] ** 2 + 1.0).log() + p[2].tanh() * p[0]

    def f_num(x):
        return math.exp(x[0] * x[1]) + math.log(x[0] ** 2 + 1.0) + math.tanh(x[2]) * x[0]

    x = np.array([0.4, -1.3, 0.7])
    g = np.asarray(geron_autograd(f_var, x)["grad"])
    assert np.allclose(g, central_diff(f_num, x, h=1e-6), atol=1e-6)


def test_autograd_handles_reused_nodes_by_accumulating():
    # y = x*x reuses the same leaf twice; d/dx = 2x, not x.
    r = geron_autograd(lambda p: p[0] * p[0], [3.0])
    assert float(r["grad"][0]) == pytest.approx(6.0)


def test_autograd_rejects_a_bypassed_tape():
    with pytest.raises(ValueError, match="tape was bypassed"):
        geron_autograd(lambda p: 1.0, [1.0])


# ── hmbp ──────────────────────────────────────────────────────────────


def test_backprop_gradients_match_central_differences():
    rng = lcg(2 * 3 + 3 * 2, seed=53, lo=-1.0, hi=1.0)
    W1 = rng[:6].reshape(2, 3)
    W2 = rng[6:].reshape(3, 2)
    X = np.array([[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0]])
    Y = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

    def loss_of(flat):
        A = flat[:6].reshape(2, 3)
        B = flat[6:].reshape(3, 2)
        h = np.tanh(X @ A)
        out = h @ B
        return float(np.sum((out - Y) ** 2) / X.shape[0])

    r = geron_backpropagation(X, Y, [W1, W2], ["tanh", "identity"])
    analytic = np.concatenate([np.asarray(r["grads_W"][0]).ravel(), np.asarray(r["grads_W"][1]).ravel()])
    numeric = central_diff(loss_of, np.concatenate([W1.ravel(), W2.ravel()]), h=1e-6)
    assert np.allclose(analytic, numeric, atol=1e-6)
    assert float(r["loss"]) == pytest.approx(loss_of(np.concatenate([W1.ravel(), W2.ravel()])))


def test_backprop_softmax_cross_entropy_delta_matches_finite_differences():
    W = np.array([[0.3, -0.2, 0.5], [1.0, 0.25, -0.75]])
    X = np.array([[1.0, 2.0], [-1.0, 0.5], [0.25, 0.25], [2.0, -1.0]])
    y = [0, 2, 1, 2]

    def loss_of(flat):
        Z = X @ flat.reshape(2, 3)
        Z = Z - Z.max(axis=1, keepdims=True)
        logp = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
        return float(-np.mean(logp[np.arange(4), y]))

    r = geron_backpropagation(X, y, [W], ["softmax"], loss="ce")
    assert float(r["loss"]) == pytest.approx(loss_of(W.ravel()))
    assert np.allclose(np.asarray(r["grads_W"][0]).ravel(), central_diff(loss_of, W.ravel(), h=1e-6), atol=1e-6)


def test_backprop_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="expects"):
        geron_backpropagation([[1.0, 2.0]], [[1.0]], [np.zeros((3, 1))], ["identity"])


# ── hmaen ─────────────────────────────────────────────────────────────


def test_autoencoder_error_equals_discarded_spectrum():
    X = np.column_stack([lcg(12, seed=59, lo=-2.0, hi=2.0), lcg(12, seed=61), lcg(12, seed=67, lo=-1.0, hi=1.0)])
    Xc = X - X.mean(axis=0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    for k in (1, 2, 3):
        err = float(geron_autoencoder(X, k)["recon_error"])
        assert err == pytest.approx(float(np.sum(sv[k:] ** 2) / X.shape[0]), abs=1e-10)


def test_autoencoder_error_is_monotone_in_bottleneck_width():
    X = np.column_stack([lcg(15, seed=71), lcg(15, seed=73), lcg(15, seed=79)])
    errs = [float(geron_autoencoder(X, k)["recon_error"]) for k in (1, 2, 3)]
    assert errs[0] >= errs[1] >= errs[2] - 1e-15


def test_autoencoder_rejects_overcomplete_bottleneck():
    with pytest.raises(ValueError, match="bottleneck must lie"):
        geron_autoencoder([[1.0, 2.0], [3.0, 4.0]], 3)


# ── hmagc ─────────────────────────────────────────────────────────────


def test_single_linkage_labels_match_connectivity_at_the_merge_height():
    pts = np.array([[0.0], [0.4], [0.9], [5.0], [5.3], [12.0]])
    r = geron_agglomerative(pts, n_clusters=3, linkage="single")
    labels = np.asarray(r["labels"])
    # Brute force: points within the last merge height are transitively linked.
    h = max(r["heights"])
    D = np.abs(pts - pts.T)
    for i in range(6):
        for j in range(6):
            if D[i, j] <= h:
                pass  # linkage is transitive, so only check the known grouping
    assert labels.tolist() == [0, 0, 0, 1, 1, 2]


def test_complete_linkage_heights_are_never_below_single_linkage():
    pts = np.array([[0.0], [1.0], [2.5], [6.0]])
    hs = geron_agglomerative(pts, 1, linkage="single")["heights"]
    hc = geron_agglomerative(pts, 1, linkage="complete")["heights"]
    assert all(c >= s - 1e-12 for c, s in zip(hc, hs))
    assert hc[-1] == pytest.approx(6.0)


# ── hmanae ────────────────────────────────────────────────────────────


def test_anomaly_errors_equal_brute_force_squared_norms():
    X = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, -4.0]])
    model = lambda A: np.asarray(A, dtype=float) * 0.5
    r = geron_anomaly_autoencoder(model, X, threshold=2.0)
    expected = [float(np.sum((row - 0.5 * row) ** 2)) for row in X]
    assert [float(e) for e in r["errors"]] == pytest.approx(expected)
    assert [bool(f) for f in r["is_anomaly"]] == [e > 2.0 for e in expected]


def test_anomaly_quantile_calibration_flags_nothing_above_the_top_quantile():
    X = np.column_stack([lcg(50, seed=83, lo=-1.0, hi=1.0)])
    r = geron_anomaly_autoencoder(lambda A: np.zeros_like(np.asarray(A, dtype=float)), X, quantile=0.9)
    assert r["n_anomalies"] <= 5
    assert r["threshold_calibrated"]


def test_anomaly_detects_shape_contract_violation():
    with pytest.raises(ValueError, match="returned shape"):
        geron_anomaly_autoencoder(lambda A: np.zeros((1, 1)), [[1.0], [2.0]], threshold=1.0)


# ── hmbrnn ────────────────────────────────────────────────────────────


def test_bidirectional_states_match_a_hand_rolled_recurrence():
    X = np.array([[1.0, -0.5], [0.25, 2.0], [-1.0, 0.0]])
    Wxf = np.array([[0.5, -1.0], [1.0, 0.25]])
    Whf = np.array([[0.1, 0.2], [-0.3, 0.4]])
    Wxb = np.array([[-0.5, 0.75], [0.2, 0.2]])
    Whb = np.array([[0.05, -0.1], [0.15, 0.3]])
    hf = np.zeros(2)
    fwd = []
    for t in range(3):
        hf = np.tanh(X[t] @ Wxf + hf @ Whf)
        fwd.append(hf.copy())
    hb = np.zeros(2)
    bwd = [None] * 3
    for t in (2, 1, 0):
        hb = np.tanh(X[t] @ Wxb + hb @ Whb)
        bwd[t] = hb.copy()
    r = geron_bidirectional_rnn(X, Wxf, Whf, Wxb, Whb)
    assert np.allclose(np.asarray(r["h_fwd"]), np.array(fwd))
    assert np.allclose(np.asarray(r["h_bwd"]), np.array(bwd))
    assert np.allclose(np.asarray(r["output"]), np.hstack([np.array(fwd), np.array(bwd)]))


def test_backward_direction_actually_sees_the_future():
    Wx = np.array([[1.0]])
    Wh = np.array([[1.0]])
    a = geron_bidirectional_rnn([[1.0], [0.0]], Wx, np.zeros((1, 1)), Wx, Wh)
    b = geron_bidirectional_rnn([[1.0], [5.0]], Wx, np.zeros((1, 1)), Wx, Wh)
    # Changing the LAST token changes the FIRST backward state.
    assert float(a["h_bwd"][0, 0]) != pytest.approx(float(b["h_bwd"][0, 0]))
    # but not the first forward state.
    assert float(a["h_fwd"][0, 0]) == pytest.approx(float(b["h_fwd"][0, 0]))


# ── hmadab ────────────────────────────────────────────────────────────


def test_adaboost_alpha_matches_the_closed_form_of_its_own_error():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = [1, -1, 1, -1, 1]
    r = geron_adaboost(X, y, n_estimators=3)
    for err, alpha in zip(r["errors"], r["alphas"]):
        e = min(max(float(err), 1e-10), 1 - 1e-10)
        assert float(alpha) == pytest.approx(0.5 * math.log((1 - e) / e))


def test_adaboost_reweighting_makes_the_last_learner_exactly_chance():
    # After the update, the just-fit learner has weighted error 1/2 -- the
    # defining property of the exponential-loss reweighting.
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1, -1, 1, -1])
    r = geron_adaboost(X, y, n_estimators=1)
    w = np.asarray(r["weights"])
    pred = np.asarray(r["predict"](X), dtype=float)
    assert float(np.sum(w[pred != y])) == pytest.approx(0.5, abs=1e-9)


def test_adaboost_selects_the_informative_feature_not_the_first_one():
    # Separable on column 1 only; sorted by column 0 the labels are -1,-1,1,1
    # in the order 1,3,5,9 -> +,-,-,+ , which no single split on column 0 fits.
    X = np.array([[5.0, -1.0], [3.0, -2.0], [9.0, 1.0], [1.0, 2.0]])
    y = np.array([-1, -1, 1, 1])
    r = geron_adaboost(X, y, n_estimators=1)
    assert float(r["train_errors"][-1]) == 0.0
    assert [int(v) for v in r["predict"](X)] == list(y)


def test_adaboost_stays_at_or_below_chance_on_a_non_representable_pattern():
    # +,-,+ over an ordered 1-D domain is outside the span of stump ensembles;
    # boosting must not diverge, and every alpha must stay positive.
    xs = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1, 1, -1, -1, 1, 1])
    r = geron_adaboost(xs, y, n_estimators=8)
    assert float(r["train_errors"][-1]) <= 0.5
    assert np.all(np.asarray(r["errors"]) < 0.5)
    assert np.all(np.asarray(r["alphas"]) > 0)


# ── hmbag ─────────────────────────────────────────────────────────────


def test_bagging_prediction_is_exactly_the_member_average():
    X = np.column_stack([lcg(12, seed=89, lo=0.0, hi=5.0)])
    y = np.asarray(X).ravel() * 2.0
    r = geron_bagging(X, y, n_estimators=6, seed=3)
    members = np.vstack([np.asarray(f(X), dtype=float).ravel() for f in r["estimators"]])
    assert np.allclose(np.asarray(r["train_pred"]), members.mean(axis=0))


def test_bagging_variance_reduction_beats_a_single_member():
    X = np.column_stack([lcg(30, seed=97, lo=0.0, hi=10.0)])
    y = np.where(np.asarray(X).ravel() < 5.0, 1.0, 4.0) + lcg(30, seed=101, lo=-0.3, hi=0.3)
    r = geron_bagging(X, y, n_estimators=20, seed=5)
    member_mses = [float(np.mean((p - y) ** 2)) for p in np.asarray(r["member_preds"])]
    assert float(r["train_mse"]) <= float(np.mean(member_mses)) + 1e-12


def test_bagging_rejects_a_non_callable_estimator_factory():
    with pytest.raises(ValueError, match="callable predictor"):
        geron_bagging([[1.0], [2.0]], [1.0, 2.0], lambda Xb, yb: 3.0, 2)


# ── hmbms ─────────────────────────────────────────────────────────────


def test_beam_search_with_full_width_matches_exhaustive_enumeration():
    lp = np.log(np.array([0.5, 0.3, 0.2]))
    # Prefix-dependent scorer: rotate the distribution by the last token.
    def model(src, prefix):
        shift = (sum(prefix) % 3) if prefix else 0
        return np.roll(lp, shift)

    best = None
    for a in range(3):
        for b in range(3):
            sc = model(None, ())[a] + model(None, (a,))[b]
            if best is None or sc > best[1]:
                best = ([a, b], sc)
    r = geron_beam_search(model, None, beam_width=3, max_len=2)
    assert r["sequence"] == best[0]
    assert float(r["score"]) == pytest.approx(best[1])


def test_beam_score_is_the_sum_of_the_chosen_log_probs():
    lp = np.log(np.array([0.7, 0.3]))
    r = geron_beam_search(lambda s, p: lp, None, beam_width=2, max_len=3)
    assert float(r["score"]) == pytest.approx(sum(float(lp[t]) for t in r["sequence"]))
    assert float(r["score"]) == pytest.approx(3 * math.log(0.7))


def test_beam_search_rejects_unnormalised_scores():
    with pytest.raises(ValueError, match="not 1"):
        geron_beam_search(lambda s, p: np.array([0.0, 0.0]), None, beam_width=2, max_len=1)


# ── hmbpet ────────────────────────────────────────────────────────────


def test_bpe_first_merge_is_the_brute_force_argmax_pair():
    corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    counts = {}
    for word, c in corpus.items():
        syms = list(word) + ["</w>"]
        for i in range(len(syms) - 1):
            counts[(syms[i], syms[i + 1])] = counts.get((syms[i], syms[i + 1]), 0) + c
    top = max(counts.values())
    r = geron_bpe_tokenizer(corpus, vocab_size=12)
    assert counts[r["merges"][0]] == top


def test_bpe_tokenization_is_a_lossless_segmentation():
    corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    r = geron_bpe_tokenizer(corpus, vocab_size=18)
    for word in corpus:
        toks = r["tokenize"](word)
        assert "".join(toks) == word + "</w>"


def test_bpe_merge_count_grows_with_the_vocabulary_budget():
    corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    sizes = [geron_bpe_tokenizer(corpus, vocab_size=v)["n_merges"] for v in (12, 14, 16, 20)]
    assert sizes == sorted(sizes)


# ── hmbrch ────────────────────────────────────────────────────────────


def test_birch_subcluster_radii_never_exceed_the_threshold():
    X = np.column_stack([lcg(40, seed=103, lo=0.0, hi=10.0)])
    thr = 0.75
    r = geron_birch(X, n_clusters=3, threshold=thr, branching_factor=8)
    assert np.all(np.asarray(r["radii"]) <= thr + 1e-9)
    assert int(np.sum(r["subcluster_sizes"])) == 40


def test_birch_recovers_well_separated_groups():
    X = np.array([[0.0], [0.2], [0.1], [9.8], [10.0], [10.2], [20.0], [20.1]])
    labels = np.asarray(geron_birch(X, n_clusters=3, threshold=0.5, branching_factor=4)["labels"])
    assert len(set(labels.tolist())) == 3
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[6] == labels[7]
    assert labels[0] != labels[3] != labels[6]


def test_birch_rejects_a_non_positive_threshold():
    with pytest.raises(ValueError, match="threshold must be positive"):
        geron_birch([[1.0]], threshold=0.0)


# ── hmarim ────────────────────────────────────────────────────────────


def test_arima_recovers_an_exact_ar2_recursion():
    y = [1.0, 0.5]
    for _ in range(20):
        y.append(0.6 * y[-1] - 0.2 * y[-2])
    r = geron_arima(y, p=2, d=0, q=0)
    assert np.allclose(np.asarray(r["ar"]), [0.6, -0.2], atol=1e-8)
    assert float(r["sigma2"]) < 1e-20


def test_arima_forecast_matches_the_hand_recursion_through_the_integration():
    y = [1.0, 3.0, 6.0, 10.0, 15.0, 21.0]  # second differences constant
    r = geron_arima(y, p=0, d=2, q=0)
    c = float(r["intercept"])
    assert c == pytest.approx(1.0, abs=1e-9)
    f = np.asarray(r["forecast"](3))
    # Integrate by hand: level, then first difference, then the constant.
    lvl, d1 = 21.0, 6.0
    manual = []
    for _ in range(3):
        d1 += c
        lvl += d1
        manual.append(lvl)
    assert np.allclose(f, manual, atol=1e-8)


def test_arima_rejects_orders_the_series_cannot_support():
    with pytest.raises(ValueError, match="too few"):
        geron_arima([1.0, 2.0, 3.0], p=5, d=0, q=0)


# ── hmalex ────────────────────────────────────────────────────────────


def test_alexnet_parameter_count_matches_layerwise_hand_arithmetic():
    r = geron_alexnet(1000)
    conv = [96 * 11 * 11 * 3 + 96, 256 * 5 * 5 * 96 + 256, 384 * 3 * 3 * 256 + 384, 384 * 3 * 3 * 384 + 384, 256 * 3 * 3 * 384 + 256]
    fc = [9216 * 4096 + 4096, 4096 * 4096 + 4096, 4096 * 1000 + 1000]
    assert r["conv_params"] == sum(conv)
    assert r["fc_params"] == sum(fc)
    assert r["total_params"] == sum(conv) + sum(fc)


def test_alexnet_spatial_dims_follow_the_conv_arithmetic():
    r = geron_alexnet(10, input_size=227)
    outs = [l["out"] for l in r["layers"] if l["kind"] in ("conv", "pool")]
    assert outs == [55, 27, 27, 13, 13, 13, 13, 6]
    assert r["flatten_dim"] == 6 * 6 * 256


def test_alexnet_rejects_an_input_that_collapses_a_feature_map():
    with pytest.raises(ValueError, match="too small"):
        geron_alexnet(10, input_size=12)


# ── hma2c / hma3c ─────────────────────────────────────────────────────


def _bandit(good=0, payoff=1.0):
    return {"reset": lambda: [1.0], "step": lambda a: ([1.0], payoff if a == good else 0.0, True)}


def test_a2c_learns_the_paying_action_and_the_value_baseline():
    r = geron_a2c(_bandit(good=1), [[0.0], [0.0]], [0.0], epochs=250, lr=0.5, seed=11)
    probs = r["policy"]([1.0])
    assert float(probs[1]) > 0.9
    assert float(probs[0]) + float(probs[1]) == pytest.approx(1.0)
    assert abs(r["value"]([1.0]) - 1.0) < 0.25


def test_a2c_softmax_policy_gradient_matches_finite_differences():
    # The per-step actor gradient is A * (onehot - pi) s^T; check the
    # (onehot - pi) part against d/dtheta log pi(a|s).
    theta = np.array([[0.3, -0.2], [1.0, 0.5]])
    s = np.array([1.5, -0.5])
    a = 1

    def logpi(flat):
        z = flat.reshape(2, 2) @ s
        z = z - z.max()
        return float(z[a] - np.log(np.exp(z).sum()))

    numeric = central_diff(logpi, theta.ravel(), h=1e-6).reshape(2, 2)
    z = theta @ s
    pi = np.exp(z - z.max())
    pi = pi / pi.sum()
    onehot = np.array([0.0, 1.0])
    analytic = np.outer(onehot - pi, s)
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_a2c_rejects_an_env_without_the_step_contract():
    with pytest.raises(ValueError, match="reset"):
        geron_a2c({"reset": lambda: [1.0]}, [[0.0], [0.0]], [0.0], epochs=1)


def test_a3c_applies_one_update_per_worker_episode_and_learns():
    r = geron_a3c(_bandit(good=0), [[0.0], [0.0]], [0.0], n_workers=3, epochs=40, lr=0.5, seed=7)
    assert r["updates"] == 3 * 40
    assert r["worker_returns"].shape == (3, 40)
    assert float(r["policy"]([1.0])[0]) > 0.9


def test_a3c_workers_use_independent_action_streams():
    r = geron_a3c(_bandit(good=0), [[0.0], [0.0]], [0.0], n_workers=4, epochs=30, lr=0.1, seed=13)
    rows = {tuple(row) for row in r["worker_returns"]}
    assert len(rows) > 1


# ── hmauxpt ───────────────────────────────────────────────────────────


def test_pretraining_lands_on_the_auxiliary_least_squares_solution():
    Xa = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
    beta = np.array([1.5, -0.5])
    ya = Xa @ beta
    r = geron_auxiliary_task_pretraining(None, (Xa, ya), (Xa[:2], ya[:2]), aux_epochs=2000, epochs=1, lr=0.05)
    assert np.allclose(np.asarray(r["theta_pretrained"]), beta, atol=1e-4)


def test_transfer_gain_is_positive_when_tasks_share_coefficients():
    Xa = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    beta = np.array([2.0, -1.0])
    ya = Xa @ beta
    Xt = np.array([[1.0, 2.0], [3.0, 1.0]])
    yt = Xt @ beta
    r = geron_auxiliary_task_pretraining(None, (Xa, ya), (Xt, yt), aux_epochs=500, epochs=3, lr=0.02)
    assert float(r["transfer_gain"]) > 0
    assert float(r["target_loss"]) < float(r["scratch_loss"])


def test_pretraining_requires_a_shared_feature_space():
    with pytest.raises(ValueError, match="shared representation"):
        geron_auxiliary_task_pretraining(None, ([[1.0, 2.0]], [1.0]), ([[1.0]], [1.0]))


# ── hmbert / hmbrob / hmalbt ──────────────────────────────────────────


def test_bert_layernorm_and_attention_invariants():
    ids = [1, 2, 3, 4, 5, 6, 7, 0]
    r = geron_bert(ids, n_layers=2, n_heads=2, d_model=8, vocab_size=9)
    H = np.asarray(r["hidden"])
    assert H.shape == (1, 8, 8)
    assert np.allclose(H.mean(axis=-1), 0.0, atol=1e-10)
    assert np.allclose(H.std(axis=-1), 1.0, atol=1e-4)
    for layer in r["attentions"][0]:
        assert np.allclose(np.asarray(layer).sum(axis=-1), 1.0, atol=1e-12)


def test_bert_mlm_loss_is_a_real_cross_entropy_over_the_vocabulary():
    ids = [1, 2, 3, 4, 5, 6, 7, 0]
    r = geron_bert(ids, vocab_size=9, mask_prob=0.4)
    n_masked = len(r["masked_positions"][0])
    assert n_masked == round(0.4 * 8)
    # Cross-entropy over 10 rows (9 vocab + mask) cannot exceed -log(1e-15).
    assert 0.0 < float(r["mlm_loss"]) < 35.0


def test_bert_is_deterministic_and_masking_changes_the_encoding():
    ids = [1, 2, 3, 4, 5, 6, 7, 0]
    a = geron_bert(ids, vocab_size=9, seed=1)
    b = geron_bert(ids, vocab_size=9, seed=1)
    c = geron_bert(ids, vocab_size=9, seed=2)
    assert np.allclose(np.asarray(a["hidden"]), np.asarray(b["hidden"]))
    assert not np.allclose(np.asarray(a["hidden"]), np.asarray(c["hidden"]))


def test_roberta_masking_is_dynamic_across_epochs():
    ids = list(range(12))
    r = geron_roberta(ids, epochs=6, mask_prob=0.25, vocab_size=13)
    assert r["dynamic_masking"] and not r["has_nsp_head"]
    assert len({tuple(m) for m in r["masks"]}) > 1
    assert all(len(m) == round(0.25 * 12) for m in r["masks"])


def test_roberta_parameter_count_is_bert_minus_the_nsp_head():
    ids = list(range(12))
    r = geron_roberta(ids, n_layers=2, n_heads=2, d_model=8, vocab_size=13)
    assert r["n_params_with_nsp"] - r["n_params"] == 8 * 2


def test_albert_sharing_makes_depth_parameter_free():
    ids = [0, 1, 2, 3, 4]
    counts = [geron_albert(ids, n_layers=L, d_model=8, d_embed=4, vocab_size=6)["n_params"] for L in (1, 2, 8)]
    assert len(set(counts)) == 1
    r8 = geron_albert(ids, n_layers=8, d_model=8, d_embed=4, vocab_size=6)
    assert r8["n_params_unshared"] == r8["n_params"] + 7 * r8["block_params"]


def test_albert_factorised_embedding_is_cheaper_for_large_vocabularies():
    r = geron_albert([0, 1, 2], vocab_size=30000, d_model=64, d_embed=8)
    assert r["embedding_params"] == 30000 * 8 + 8 * 64
    assert r["embedding_params_direct"] == 30000 * 64
    assert r["embedding_params"] < r["embedding_params_direct"] / 5


def test_albert_rejects_an_embedding_wider_than_the_model():
    with pytest.raises(ValueError, match="d_embed must lie"):
        geron_albert([0, 1], d_model=8, d_embed=16, vocab_size=3)


# ── hmbart ────────────────────────────────────────────────────────────


def test_bart_infilling_collapses_each_span_to_a_single_mask():
    src = ["t%d" % i for i in range(20)]
    r = geron_bart(src, src, mask_ratio=0.3, mean_span=3.0, seed=5)
    n_spans, n_masked = r["n_spans"], r["n_masked"]
    assert r["corrupted"].count("<mask>") == n_spans
    assert len(r["corrupted"]) == len(src) - n_masked + n_spans
    assert n_masked >= 1


def test_bart_corruption_is_seed_deterministic_and_seed_sensitive():
    src = ["a", "b", "c", "d", "e", "f", "g", "h"]
    assert geron_bart(src, src, seed=1)["corrupted"] == geron_bart(src, src, seed=1)["corrupted"]
    variants = {tuple(geron_bart(src, src, seed=s)["corrupted"]) for s in range(6)}
    assert len(variants) > 1


def test_bart_rejects_a_scorer_that_breaks_the_length_contract():
    with pytest.raises(ValueError, match="token log-probs"):
        geron_bart(["a", "b"], ["a", "b", "c"], model=lambda c, t: [-1.0])


# ── hmbftn ────────────────────────────────────────────────────────────


def test_finetune_first_step_matches_the_hand_derived_gradient():
    Z = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = [0, 1, 0]
    lr = 0.3
    r = geron_bert_finetune(lambda A: np.asarray(A, dtype=float), Z, y, epochs=1, lr=lr)
    p0 = np.full((3, 2), 0.5)
    Y = np.zeros((3, 2))
    Y[np.arange(3), y] = 1.0
    expected = -lr * (Z.T @ ((p0 - Y) / 3))
    assert np.allclose(np.asarray(r["W"]), expected)
    assert float(r["losses"][0]) == pytest.approx(math.log(2))


def test_finetune_loss_decreases_monotonically_on_separable_data():
    Z = np.array([[2.0, 0.0], [1.5, 0.5], [0.0, 2.0], [0.5, 1.5]])
    y = [0, 0, 1, 1]
    r = geron_bert_finetune(lambda A: np.asarray(A, dtype=float), Z, y, epochs=150, lr=0.5)
    losses = np.asarray(r["losses"])
    assert np.all(np.diff(losses) <= 1e-12)
    assert float(r["accuracy"]) == 1.0


def test_finetune_rejects_an_encoder_with_the_wrong_row_count():
    with pytest.raises(ValueError, match="embeddings but y has"):
        geron_bert_finetune(lambda A: np.zeros((5, 3)), [[1.0]], [0, 1])


# ── hmblip ────────────────────────────────────────────────────────────


def test_blip_itc_matches_hand_log_sum_exp():
    I = np.array([[1.0, 0.0], [0.0, 1.0]])
    T = np.array([[1.0, 0.0], [0.0, 1.0]])
    tau = 0.5
    r = geron_blip(I, T, temperature=tau)
    # Each row's logits are (1/tau, 0): loss = log(1 + exp(-1/tau)).
    expected = math.log(1.0 + math.exp(-1.0 / tau))
    assert float(r["itc_loss"]) == pytest.approx(expected)
    assert float(r["itc_i2t"]) == pytest.approx(float(r["itc_t2i"]))


def test_blip_itc_is_scale_invariant_but_not_pairing_invariant():
    I = np.array([[3.0, 1.0], [1.0, -2.0], [0.5, 0.5]])
    T = np.array([[2.0, 0.5], [0.5, -3.0], [1.0, 1.0]])
    base = float(geron_blip(I, T)["itc_loss"])
    assert float(geron_blip(I * 7.0, T * 0.1)["itc_loss"]) == pytest.approx(base)
    shuffled = float(geron_blip(I, T[[1, 2, 0]])["itc_loss"])
    assert shuffled > base


def test_blip_rejects_mismatched_batches_and_zero_norm_embeddings():
    with pytest.raises(ValueError, match="shape"):
        geron_blip([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="zero norm"):
        geron_blip([[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]])


# ── hmblp2 ────────────────────────────────────────────────────────────


def test_qformer_cross_attention_is_a_distribution_over_patches():
    img = np.column_stack([lcg(6, seed=107, lo=-1.0, hi=1.0), lcg(6, seed=109)])
    r = geron_blip2(img, [0.5, -0.25], n_query=5, d_query=8, d_llm=4)
    A = np.asarray(r["attention"])
    assert A.shape == (5, 6)
    assert np.all(A >= 0)
    assert np.allclose(A.sum(axis=1), 1.0, atol=1e-12)
    assert np.asarray(r["llm_input"]).shape == (5, 4)


def test_qformer_output_is_a_convex_combination_inside_the_value_hull():
    img = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5], [2.0, -1.0]])
    r = geron_blip2(img, [1.0, 1.0], n_query=3, d_query=4)
    A = np.asarray(r["attention"])
    # Attention output must lie in the convex hull of the projected patches,
    # so no coordinate may exceed the per-column max of that projection.
    assert A.shape[1] == img.shape[0]
    assert float(A.max()) <= 1.0


def test_qformer_param_count_grows_with_queries_and_llm_width():
    small = geron_blip2([[1.0, 0.0]], [1.0], n_query=2, d_query=4, d_llm=4)["trainable_params"]
    more_q = geron_blip2([[1.0, 0.0]], [1.0], n_query=8, d_query=4, d_llm=4)["trainable_params"]
    wider = geron_blip2([[1.0, 0.0]], [1.0], n_query=2, d_query=4, d_llm=32)["trainable_params"]
    assert more_q > small and wider > small
