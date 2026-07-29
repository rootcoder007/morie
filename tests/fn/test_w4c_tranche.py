# morie.fn -- test file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent-route checks for the w4c tranche (Geron, Hands-On ML).

Every assertion is derived from the mathematics rather than from a
recorded run: gradients against central finite differences, metrics
against brute-force counting, optimiser steps against hand-evaluated
recursions, losses against hand log-sums, and bounds/monotonicity that a
mean-of-inputs stub cannot satisfy.
"""

import math

import numpy as np
import pytest


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def lcg(n, seed=1, lo=0.0, hi=1.0):
    """Deterministic uniforms, the same stream the modules use."""
    s = int(seed) % 2**32
    out = np.empty(n)
    for i in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = lo + (s + 0.5) / 2**32 * (hi - lo)
    return out


def fd_grad(f, x, h=1e-6):
    """Central-difference gradient, the independent route for every analytic one."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        up, dn = x.copy(), x.copy()
        up.flat[i] += h
        dn.flat[i] -= h
        g.flat[i] = (f(up) - f(dn)) / (2 * h)
    return g


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def test_hmrms_matches_a_hand_loop_and_is_not_a_mean():
    from morie.fn.hmrms import geron_rmse

    a = [1.0, 2.0, 3.0, 4.0]
    b = [1.5, 1.0, 4.5, 4.0]
    want = math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)) / len(a))
    got = float(geron_rmse(a, b)["rmse"])
    assert abs(got - want) < 1e-12
    # A stub returning the mean of its inputs would land on 2.5.
    assert abs(got - float(np.mean(a))) > 0.5


def test_hmrms_rmse_dominates_mae():
    from morie.fn.hmrms import geron_rmse

    r = geron_rmse([0.0, 0.0, 0.0], [0.0, 0.0, 3.0])
    assert r["rmse"] >= r["mae"] - 1e-15
    assert abs(float(r["mse"]) - 3.0) < 1e-12


def test_hmpre_counts_by_brute_force():
    from morie.fn.hmpre import geron_precision

    yt = [1, 0, 1, 1, 0, 1, 0]
    yp = [1, 1, 1, 0, 0, 1, 1]
    tp = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(yt, yp) if a != 1 and b == 1)
    r = geron_precision(yt, yp)
    assert (r["tp"], r["fp"]) == (tp, fp)
    assert abs(float(r["precision"]) - tp / (tp + fp)) < 1e-15
    assert abs(float(r["precision"]) - float(np.mean(yt))) > 1e-6


def test_hmpre_undefined_without_positive_predictions():
    from morie.fn.hmpre import geron_precision

    with pytest.raises(ValueError, match="undefined"):
        geron_precision([1, 0], [0, 0])


def test_hmrec_counts_by_brute_force():
    from morie.fn.hmrec import geron_recall

    yt = [1, 1, 1, 0, 0]
    yp = [1, 0, 1, 1, 0]
    tp = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 1)
    fn = sum(1 for a, b in zip(yt, yp) if a == 1 and b != 1)
    r = geron_recall(yt, yp)
    assert (r["tp"], r["fn"]) == (tp, fn)
    assert abs(float(r["recall"]) - tp / (tp + fn)) < 1e-15


def test_hmrec_undefined_without_positives():
    from morie.fn.hmrec import geron_recall

    with pytest.raises(ValueError, match="undefined"):
        geron_recall([0, 0], [1, 0])


def test_hmprc_average_precision_by_hand_sweep():
    from morie.fn.hmprc import geron_precision_recall_curve

    y = [0, 1, 1, 0]
    s = [0.1, 0.9, 0.4, 0.3]
    order = sorted(range(4), key=lambda i: -s[i])
    tp = 0
    P = sum(y)
    prev_rec = 0.0
    ap = 0.0
    for k, i in enumerate(order, start=1):
        tp += y[i]
        rec = tp / P
        ap += (rec - prev_rec) * (tp / k)
        prev_rec = rec
    r = geron_precision_recall_curve(y, s)
    assert abs(float(r["average_precision"]) - ap) < 1e-12
    assert np.all(r["precision"] <= 1.0) and np.all(r["recall"] <= 1.0)


def test_hmprc_needs_a_positive():
    from morie.fn.hmprc import geron_precision_recall_curve

    with pytest.raises(ValueError):
        geron_precision_recall_curve([0, 0], [0.1, 0.2])


def test_hmroc_curve_matches_threshold_counting():
    from morie.fn.hmroc import geron_roc_curve

    y = np.array([0, 1, 0, 1, 1])
    s = np.array([0.2, 0.9, 0.4, 0.35, 0.8])
    r = geron_roc_curve(y, s)
    P, N = int(y.sum()), int((1 - y).sum())
    for thr, tpr, fpr in zip(r["thresholds"], r["tpr"], r["fpr"]):
        pred = s >= thr
        assert abs(tpr - np.sum(pred & (y == 1)) / P) < 1e-12
        assert abs(fpr - np.sum(pred & (y == 0)) / N) < 1e-12


def test_hmroc_auc_equals_pairwise_win_rate():
    from morie.fn.hmroc import geron_roc_curve

    y = np.array([0, 1, 0, 1, 1])
    s = np.array([0.2, 0.9, 0.4, 0.35, 0.8])
    wins = sum(
        1.0 if s[i] > s[j] else 0.5 if s[i] == s[j] else 0.0
        for i in np.where(y == 1)[0]
        for j in np.where(y == 0)[0]
    )
    want = wins / (int(y.sum()) * int((1 - y).sum()))
    r = geron_roc_curve(y, s)
    assert abs(float(r["auc"]) - want) < 1e-12
    assert abs(float(r["auc_trapezoid"]) - want) < 1e-12


# --------------------------------------------------------------------------
# activations
# --------------------------------------------------------------------------
def test_hmrelu_output_and_subgradient():
    from morie.fn.hmrelu import geron_relu

    z = np.array([-3.0, -0.5, 0.5, 4.0])
    r = geron_relu(z)
    assert np.allclose(r["a"], np.maximum(z, 0.0))
    for i, zi in enumerate(z):
        if abs(zi) > 1e-3:
            fd = (max(zi + 1e-6, 0) - max(zi - 1e-6, 0)) / 2e-6
            assert abs(float(r["gradient"][i]) - fd) < 1e-6
    assert abs(float(r["dead_fraction"]) - 0.5) < 1e-12


def test_hmrelu_leaky_slope_is_used():
    from morie.fn.hmrelu import geron_relu

    r = geron_relu([-4.0], leaky=0.25)
    assert abs(float(r["a"][0]) + 1.0) < 1e-12
    with pytest.raises(ValueError):
        geron_relu([1.0], leaky=-0.1)


def test_hmprel_gradients_match_finite_differences():
    from morie.fn.hmprel import geron_prelu

    z = np.array([[-2.0, 3.0], [1.0, -0.5]])
    alpha = np.array([0.3, 0.7])

    def loss_z(v):
        v = v.reshape(z.shape)
        return float(np.sum(np.where(v < 0, alpha * v, v)))

    def loss_a(a):
        return float(np.sum(np.where(z < 0, a * z, z)))

    r = geron_prelu(z, alpha)
    assert np.allclose(r["grad_z"].ravel(), fd_grad(loss_z, z.ravel()), atol=1e-6)
    assert np.allclose(np.asarray(r["grad_alpha"]), fd_grad(loss_a, alpha), atol=1e-6)


def test_hmprel_channel_count_is_enforced():
    from morie.fn.hmprel import geron_prelu

    with pytest.raises(ValueError, match="channels"):
        geron_prelu([[1.0, 2.0, 3.0]], [0.1, 0.2])


# --------------------------------------------------------------------------
# linear models
# --------------------------------------------------------------------------
def test_hmmsec_cost_and_gradient():
    from morie.fn.hmmsec import geron_linreg_mse_cost

    X = np.array([[1.0, 2.0], [1.0, -1.0], [1.0, 0.5]])
    y = np.array([3.0, -1.0, 1.0])
    th = np.array([0.3, 1.7])
    want = float(np.mean([(X[i] @ th - y[i]) ** 2 for i in range(3)]))
    r = geron_linreg_mse_cost(X, y, th)
    assert abs(float(r["cost"]) - want) < 1e-12
    fd = fd_grad(lambda t: float(np.mean((X @ t - y) ** 2)), th)
    assert np.allclose(r["gradient"], fd, atol=1e-6)


def test_hmneq_agrees_with_least_squares_and_refuses_collinearity():
    from morie.fn.hmneq import geron_normal_equation

    X = np.column_stack([np.ones(5), lcg(5, 3, -2, 2), lcg(5, 9, -1, 1)])
    y = X @ np.array([0.5, -1.5, 2.0]) + lcg(5, 21, -0.05, 0.05)
    want = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(geron_normal_equation(X, y)["theta"], want, atol=1e-9)
    with pytest.raises(ValueError, match="collinear"):
        geron_normal_equation(np.column_stack([np.ones(3), np.ones(3)]), [1.0, 2.0, 3.0])


def test_hmridg_gradient_matches_finite_differences():
    from morie.fn.hmridg import geron_ridge_cost

    X = np.array([[1.0, 2.0], [1.0, -1.0], [1.0, 3.0]])
    y = np.array([1.0, 0.0, 2.0])
    th = np.array([0.4, -0.7])
    a = 0.9

    def J(t):
        return float(np.mean((X @ t - y) ** 2) + 0.5 * a * t[1] ** 2)

    r = geron_ridge_cost(X, y, th, a)
    assert abs(float(r["cost"]) - J(th)) < 1e-12
    assert np.allclose(r["gradient"], fd_grad(J, th), atol=1e-6)


def test_hmridn_solves_the_augmented_system_and_shrinks():
    from morie.fn.hmridn import geron_ridge_normal

    X = np.column_stack([np.ones(6), lcg(6, 5, -2, 2)])
    y = X @ np.array([1.0, 2.0]) + lcg(6, 7, -0.2, 0.2)
    a = 2.0
    th = np.asarray(geron_ridge_normal(X, y, a)["theta"])
    G = X.T @ X + a * np.diag([0.0, 1.0])
    assert np.allclose(G @ th, X.T @ y, atol=1e-9)
    small = np.asarray(geron_ridge_normal(X, y, 0.0)["theta"])
    assert abs(th[1]) < abs(small[1])


def test_hmplf_count_and_values():
    from morie.fn.hmplf import geron_polynomial_features

    X = np.array([[2.0, 3.0, 5.0]])
    r = geron_polynomial_features(X, 3)
    n, d = 3, 3
    want = math.comb(n + d, d)
    assert int(r["n_output_features"]) == want
    for row, name in zip(r["powers"], r["names"]):
        assert abs(float(r["features"][0][list(r["names"]).index(name)]) - float(np.prod(X[0] ** row))) < 1e-9


def test_hmplf_rejects_degree_zero():
    from morie.fn.hmplf import geron_polynomial_features

    with pytest.raises(ValueError):
        geron_polynomial_features([[1.0]], 0)


# --------------------------------------------------------------------------
# differentiation
# --------------------------------------------------------------------------
def test_hmnmd_matches_analytic_derivatives():
    from morie.fn.hmnmd import geron_numerical_diff

    assert abs(float(geron_numerical_diff(math.exp, 1.3)["derivative"]) - math.exp(1.3)) < 1e-6
    g = geron_numerical_diff(lambda v: float(np.sin(v[0]) * v[1] ** 2), [0.7, 1.5])["derivative"]
    assert abs(g[0] - math.cos(0.7) * 1.5**2) < 1e-6
    assert abs(g[1] - math.sin(0.7) * 2 * 1.5) < 1e-6


def test_hmnmd_rejects_bad_steps_and_non_scalar_f():
    from morie.fn.hmnmd import geron_numerical_diff

    with pytest.raises(ValueError):
        geron_numerical_diff(math.exp, 1.0, h=0.0)
    with pytest.raises(ValueError, match="scalar"):
        geron_numerical_diff(lambda v: np.asarray([1.0, 2.0]), [1.0])


def test_hmrad_gradient_matches_finite_differences():
    from morie.fn.hmrad import geron_reverse_autodiff

    def f_var(v):
        return (v[0] * v[1]).tanh() + (v[0] ** 2).exp()

    def f_num(x):
        return math.tanh(x[0] * x[1]) + math.exp(x[0] ** 2)

    x = np.array([0.4, -1.1])
    got = np.asarray(geron_reverse_autodiff(f_var, x)["gradient"])
    assert np.allclose(got, fd_grad(f_num, x), atol=1e-6)


def test_hmrad_rejects_a_bypassed_tape():
    from morie.fn.hmrad import geron_reverse_autodiff

    with pytest.raises(ValueError):
        geron_reverse_autodiff(lambda v: 3.0, [1.0])


# --------------------------------------------------------------------------
# optimisers and cells
# --------------------------------------------------------------------------
def test_hmnadm_first_step_by_hand():
    from morie.fn.hmnadm import geron_nadam

    b1, b2, eta, g = 0.9, 0.999, 0.1, 1.0
    m = (1 - b1) * g
    v = (1 - b2) * g * g
    m_hat = b1 * m / (1 - b1**2) + (1 - b1) * g / (1 - b1)
    v_hat = v / (1 - b2)
    want = -eta * m_hat / (math.sqrt(v_hat) + 1e-8)
    r = geron_nadam([g], eta=eta, t=1)
    assert abs(float(r["step"][0]) - want) < 1e-12


def test_hmnadm_differs_from_plain_adam():
    from morie.fn.hmadam import geron_adam
    from morie.fn.hmnadm import geron_nadam

    a = float(geron_adam([1.0], eta=0.1, t=1)["step"][0])
    n = float(geron_nadam([1.0], eta=0.1, t=1)["step"][0])
    assert abs(a - n) > 1e-3
    with pytest.raises(ValueError):
        geron_nadam([1.0], t=0)


def test_hmnag_two_steps_match_the_recursion():
    from morie.fn.hmnag import geron_nesterov

    beta, eta = 0.9, 0.1
    theta, v = np.array([1.0]), np.array([0.0])
    for _ in range(2):
        look = theta - eta * beta * v
        v = beta * v + look  # gradient of x^2/2 is x
        theta = theta - eta * v
    r1 = geron_nesterov(lambda x: x, v=[0.0], beta=beta, eta=eta, theta=[1.0])
    r2 = geron_nesterov(lambda x: x, v=r1["v"], beta=beta, eta=eta, theta=r1["theta"])
    assert abs(float(r2["theta"][0]) - float(theta[0])) < 1e-12
    assert abs(float(r2["v"][0]) - float(v[0])) < 1e-12


def test_hmnag_requires_theta_with_a_callable():
    from morie.fn.hmnag import geron_nesterov

    with pytest.raises(ValueError, match="theta is required"):
        geron_nesterov(lambda x: x, v=[0.0])


def test_hmrnn_state_and_jacobian():
    from morie.fn.hmrnn import geron_recurrent_neuron

    Wx = np.array([[0.5, -0.2], [0.1, 0.4]])
    Wh = np.array([[0.3, 0.0], [-0.6, 0.2]])
    b = np.array([0.05, -0.1])
    x = np.array([1.0, -2.0])
    h = np.array([0.4, 0.7])
    want = np.tanh(Wx @ x + Wh @ h + b)
    r = geron_recurrent_neuron(x, h, Wx, Wh, b)
    assert np.allclose(r["h"], want)
    fd = np.column_stack([fd_grad(lambda hh: float(np.tanh(Wx @ x + Wh @ hh + b)[i]), h) for i in range(2)]).T
    assert np.allclose(r["jacobian"], fd, atol=1e-6)


def test_hmrnn_rejects_shape_mismatch():
    from morie.fn.hmrnn import geron_recurrent_neuron

    with pytest.raises(ValueError, match="Wx"):
        geron_recurrent_neuron([1.0], [0.0], [[1.0], [1.0]], [[1.0]], [0.0])


def test_hmphp_gates_by_hand_and_peephole_effect():
    from morie.fn.hmphp import geron_peephole_lstm

    sig = lambda z: 1.0 / (1.0 + math.exp(-z))
    W = {"W_x": [[0.5], [-0.5], [1.0], [0.25]], "W_h": [[0.0]] * 4, "b": [0.1, 0.2, 0.0, -0.1]}
    r = geron_peephole_lstm([2.0], [0.0], [0.5], W)
    i = sig(0.5 * 2 + 0.1)
    f = sig(-0.5 * 2 + 0.2)
    gg = math.tanh(1.0 * 2)
    c = f * 0.5 + i * gg
    o = sig(0.25 * 2 - 0.1)
    assert abs(float(r["c"][0]) - c) < 1e-12
    assert abs(float(r["h"][0]) - o * math.tanh(c)) < 1e-12
    peeped = geron_peephole_lstm([2.0], [0.0], [0.5], dict(W, p_f=[4.0]))
    assert abs(float(peeped["f"][0]) - sig(-1.0 + 0.2 + 4.0 * 0.5)) < 1e-12
    assert float(peeped["f"][0]) > float(r["f"][0])


def test_hmphp_requires_the_weight_block():
    from morie.fn.hmphp import geron_peephole_lstm

    with pytest.raises(ValueError, match="W_h"):
        geron_peephole_lstm([1.0], [0.0], [0.0], {"W_x": [[0.0]] * 4, "b": [0.0] * 4})


def test_hmpcpt_separates_and_stalls_where_it_must():
    from morie.fn.hmpcpt import geron_perceptron

    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    r = geron_perceptron(X, [0, 0, 0, 1], n_iter=30)
    w, b = np.asarray(r["w"]), float(r["bias"])
    for xi, yi in zip(X, [0, 0, 0, 1]):
        assert (np.dot(w, xi) + b >= 0) == bool(yi)
    x = geron_perceptron(X, [0, 1, 1, 0], n_iter=60)
    assert not x["converged"] and x["accuracy"] < 1.0


def test_hmpcpt_rejects_non_binary_labels():
    from morie.fn.hmpcpt import geron_perceptron

    with pytest.raises(ValueError, match="0 or 1"):
        geron_perceptron([[1.0], [2.0]], [0, 2])


def test_hmresn_identity_path_and_contract():
    from morie.fn.hmresn import geron_resnet

    x = np.array([1.0, -2.0, 3.0])
    assert np.allclose(geron_resnet(x, lambda a: a * 0.5)["y"], 1.5 * x)
    assert float(geron_resnet(x, lambda a: np.zeros_like(a))["residual_fraction"]) == 0.0
    with pytest.raises(ValueError, match="skip path"):
        geron_resnet(x, lambda a: np.zeros(2))


def test_hmql_update_and_terminal_case():
    from morie.fn.hmql import geron_q_learning

    Q = [[0.2, 0.4], [1.0, -1.0]]
    r = geron_q_learning(Q, 0, 1, 2.0, 1, 0.25, 0.5)
    target = 2.0 + 0.5 * max(Q[1])
    assert abs(float(r["target"]) - target) < 1e-12
    assert abs(float(r["new_value"]) - (0.4 + 0.25 * (target - 0.4))) < 1e-12
    assert float(geron_q_learning(Q, 0, 1, 2.0, 1, 1.0, 0.5, done=True)["new_value"]) == 2.0
    assert np.allclose(np.asarray(Q, dtype=float), [[0.2, 0.4], [1.0, -1.0]])  # not mutated


def test_hmql_validates_indices_and_rates():
    from morie.fn.hmql import geron_q_learning

    with pytest.raises(ValueError):
        geron_q_learning([[0.0, 0.0]], 5, 0, 1.0, 0, 0.5, 0.9)
    with pytest.raises(ValueError):
        geron_q_learning([[0.0, 0.0]], 0, 0, 1.0, 0, 0.0, 0.9)


def test_hmpol_entropy_and_validation():
    from morie.fn.hmpol import geron_policy

    p = [0.1, 0.2, 0.7]
    want = -sum(q * math.log(q) for q in p)
    r = geron_policy(0, [p])
    assert abs(float(r["entropy"]) - want) < 1e-12
    assert int(r["greedy_action"]) == 2
    assert 0 <= int(r["action"]) < 3
    with pytest.raises(ValueError, match="sum"):
        geron_policy(0, [[0.5, 0.2]])


def test_hmper_probabilities_and_weights_by_hand():
    from morie.fn.hmper import geron_prioritized_replay

    d = [0.5, 1.5, 2.0]
    a, b = 0.6, 0.4
    pri = [x**a for x in d]
    prob = [x / sum(pri) for x in pri]
    w = [(3 * q) ** (-b) for q in prob]
    w = [x / max(w) for x in w]
    r = geron_prioritized_replay(d, alpha=a, beta=b, eps=0.0)
    assert np.allclose(r["probabilities"], prob)
    assert np.allclose(r["weights"], w)
    assert abs(float(np.sum(r["probabilities"])) - 1.0) < 1e-12


def test_hmper_draws_are_in_range():
    from morie.fn.hmper import geron_prioritized_replay

    r = geron_prioritized_replay([{"td_error": -2.0}, {"td_error": 0.0}], batch_size=16, seed=5)
    assert r["indices"].shape == (16,)
    assert set(np.unique(r["indices"]).tolist()) <= {0, 1}


# --------------------------------------------------------------------------
# preprocessing and diagnostics
# --------------------------------------------------------------------------
def test_hmohe_is_an_indicator_matrix():
    from morie.fn.hmohe import geron_one_hot_encoding

    X = ["b", "a", "c", "a"]
    r = geron_one_hot_encoding(X)
    cats = [str(c) for c in r["categories"][0]]
    E = np.asarray(r["encoded"])
    assert np.allclose(E.sum(axis=1), 1.0)
    for i, token in enumerate(X):
        assert cats[int(np.argmax(E[i]))] == token
    assert np.asarray(geron_one_hot_encoding(X, drop_first=True)["encoded"]).shape[1] == len(cats) - 1


def test_hmohe_rejects_unseen_levels():
    from morie.fn.hmohe import geron_one_hot_encoding

    with pytest.raises(ValueError, match="not in categories"):
        geron_one_hot_encoding(["a", "q"], categories=["a", "b"])


def test_hmord_codes_follow_the_given_order():
    from morie.fn.hmord import geron_ordinal_encoding

    order = ["low", "mid", "high"]
    X = ["high", "low", "mid", "high"]
    r = geron_ordinal_encoding(X, categories=order)
    assert [int(v) for v in r["encoded"].ravel()] == [order.index(t) for t in X]
    with pytest.raises(ValueError, match="not in categories"):
        geron_ordinal_encoding(["x"], categories=order)


def test_hmovf_gap_and_early_stopping_point():
    from morie.fn.hmovf import geron_overfitting

    tr = [0.9, 0.5, 0.3, 0.15, 0.05]
    va = [1.0, 0.6, 0.4, 0.45, 0.8]
    r = geron_overfitting(tr, va)
    assert abs(float(r["gap"]) - (va[-1] - tr[-1])) < 1e-12
    assert int(r["best_epoch"]) == int(np.argmin(va))
    assert int(r["epochs_past_best"]) == len(va) - 1 - int(np.argmin(va))
    assert bool(r["overfitting"]) is True


def test_hmovf_rejects_negative_errors():
    from morie.fn.hmovf import geron_overfitting

    with pytest.raises(ValueError):
        geron_overfitting(-0.1, 0.2)


def test_hmnpl_parameter_count_by_hand():
    from morie.fn.hmnpl import geron_neurons_per_layer

    d, L, k = 7, 3, 2
    r = geron_neurons_per_layer(d, n_layers=L, n_outputs=k)
    w = int(r["width"])
    want = (d * w + w) + (L - 1) * (w * w + w) + (w * k + k)
    assert int(r["n_parameters"]) == want
    assert r["width_range"] == (d, 2 * d)
    with pytest.raises(ValueError):
        geron_neurons_per_layer(0)


def test_hmpe_matches_the_sinusoid_definition():
    from morie.fn.hmpe import geron_positional_encoding

    d, pos = 8, 5
    pe = np.asarray(geron_positional_encoding(pos, d)["pe"])
    for i in range(d // 2):
        ang = pos / 10000 ** (2 * i / d)
        assert abs(pe[2 * i] - math.sin(ang)) < 1e-12
        assert abs(pe[2 * i + 1] - math.cos(ang)) < 1e-12
    assert float(geron_positional_encoding([0, 1, 2], d)["rotation_check"]) < 1e-12


def test_hmpe_requires_an_even_width():
    from morie.fn.hmpe import geron_positional_encoding

    with pytest.raises(ValueError, match="even"):
        geron_positional_encoding(1, 5)


def test_hmpd_pads_and_preserves_same_size():
    from morie.fn.hmpd import geron_padding

    x = np.arange(9.0).reshape(3, 3)
    r = geron_padding(x, 2, 1)
    assert r["padded"].shape == (7, 5)
    assert abs(float(r["padded"].sum()) - float(x.sum())) < 1e-12
    assert np.allclose(r["padded"][2:5, 1:4], x)
    same = geron_padding(np.ones((6, 6)), kernel_size=5)
    assert same["output_shape"] == (6, 6)
    with pytest.raises(ValueError):
        geron_padding(x)


# --------------------------------------------------------------------------
# unsupervised
# --------------------------------------------------------------------------
def test_hmpcac_components_are_covariance_eigenvectors():
    from morie.fn.hmpcac import geron_principal_components

    X = np.column_stack([lcg(30, 4, -3, 3), lcg(30, 8, -1, 1)])
    X[:, 1] += 0.5 * X[:, 0]
    r = geron_principal_components(X, 2)
    S = np.cov(X, rowvar=False)
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    assert np.allclose(np.sort(np.asarray(r["explained_variance"]))[::-1], vals[order], atol=1e-9)
    v = vecs[:, order[0]]
    got = np.asarray(r["components"])[:, 0]
    assert min(np.linalg.norm(got - v), np.linalg.norm(got + v)) < 1e-8
    assert abs(float(np.var(np.asarray(r["scores"])[:, 0], ddof=1)) - float(r["explained_variance"][0])) < 1e-9


def test_hmpcav_first_component_maximises_the_quadratic_form():
    from morie.fn.hmpcav import geron_pca_variance

    X = np.column_stack([lcg(40, 11, -2, 2), lcg(40, 13, -2, 2)])
    r = geron_pca_variance(X, n_probes=200, seed=3)
    S = np.cov(X, rowvar=False)
    assert abs(float(r["top_variance"]) - float(np.max(np.linalg.eigvalsh(S)))) < 1e-9
    assert float(r["probe_max"]) <= float(r["top_variance"]) + 1e-12
    assert abs(float(r["cumulative"][-1]) - 1.0) < 1e-9


def test_hmnmf_factors_are_non_negative_and_reconstruct():
    from morie.fn.hmnmf import geron_nmf

    W0 = np.array([[1.0, 0.2], [0.5, 1.0], [0.9, 0.1]])
    H0 = np.array([[1.0, 0.3, 0.7], [0.2, 1.0, 0.4]])
    X = W0 @ H0
    r = geron_nmf(X, 2, max_iter=2000)
    assert np.all(np.asarray(r["W"]) >= 0) and np.all(np.asarray(r["H"]) >= 0)
    recon = np.asarray(r["W"]) @ np.asarray(r["H"])
    assert abs(float(np.linalg.norm(X - recon, "fro")) - float(r["reconstruction_error"])) < 1e-9
    assert float(r["relative_error"]) < 1e-3
    with pytest.raises(ValueError, match="non-negative"):
        geron_nmf([[1.0, -1.0], [1.0, 1.0]], 1)


def test_hmrpca_recovers_the_exact_leading_spectrum():
    from morie.fn.hmrpca import geron_randomized_pca

    base = lcg(20, 17, -2, 2)
    X = np.column_stack([base, 2 * base, -base]) + 1e-9
    r = geron_randomized_pca(X, 1, seed=2)
    Xc = X - X.mean(axis=0)
    exact = np.linalg.svd(Xc, compute_uv=False)
    assert abs(float(r["singular_values"][0]) - float(exact[0])) < 1e-6
    assert abs(float(r["explained_variance"][0]) - exact[0] ** 2 / (X.shape[0] - 1)) < 1e-6
    assert float(r["explained_variance_ratio"][0]) > 0.999


def test_hmopt_reachability_on_a_known_line():
    from morie.fn.hmopt import geron_optics

    X = [[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]]
    r = geron_optics(X, min_samples=2, eps_cluster=2.0)
    assert np.allclose(np.asarray(r["core_distances"]), [1, 1, 1, 1, 1, 1])
    lab = np.asarray(r["labels"])
    assert int(r["n_clusters"]) == 2
    assert lab[0] == lab[1] == lab[2] and lab[3] == lab[4] == lab[5] and lab[0] != lab[3]
    assert set(np.asarray(r["ordering"]).tolist()) == set(range(6))


def test_hmocsv_dual_constraints_hold():
    from morie.fn.hmocsv import geron_one_class_svm

    X = np.array([[0.0], [0.1], [-0.1], [0.05], [-0.05], [4.0]])
    r = geron_one_class_svm(X, nu=0.5, gamma=1.0)
    a = np.asarray(r["alpha"])
    assert abs(float(a.sum()) - 1.0) < 1e-8
    assert np.all(a >= -1e-9) and np.all(a <= float(r["C"]) + 1e-9)
    assert int(np.argmin(np.asarray(r["decision"]))) == 5
    assert float(r["outlier_fraction"]) <= 0.5 + 1e-9
    with pytest.raises(ValueError):
        geron_one_class_svm(X, nu=0.0)


def test_hmnov_ratio_matches_a_hand_gaussian():
    from morie.fn.hmnov import geron_novelty_detection

    train = np.array([[-1.0], [0.0], [1.0], [0.5], [-0.5]])
    mu = float(train.mean())
    var = float(((train - mu) ** 2).sum() / (train.size - 1)) + 1e-9
    ld = lambda x: -0.5 * ((x - mu) ** 2 / var + math.log(var) + math.log(2 * math.pi))
    ref = float(np.mean([ld(float(t[0])) for t in train]))
    r = geron_novelty_detection(train, [[0.0], [6.0]])
    assert abs(float(r["log_density"][0]) - ld(0.0)) < 1e-9
    assert abs(float(r["reference"]) - ref) < 1e-9
    assert abs(float(r["ratio"][1]) - math.exp(ld(6.0) - ref)) < 1e-9
    assert [bool(v) for v in r["is_novel"]] == [False, True]


# --------------------------------------------------------------------------
# ensembles and trees
# --------------------------------------------------------------------------
def test_hmoob_score_by_brute_force():
    from morie.fn.hmoob import geron_oob_score

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    f = lambda A: np.asarray(A, dtype=float)[:, 0] + 1.0
    bags = [[True, True, False, False], [False, False, True, True]]
    r = geron_oob_score(X, y, [(f, b) for b in bags], task="regression")
    want = np.mean([(f(X[i : i + 1])[0] - y[i]) ** 2 for i in range(4)])
    assert abs(float(r["oob_score"]) - float(want)) < 1e-12
    assert float(r["mean_oob_votes"]) == 1.0


def test_hmoob_refuses_full_coverage():
    from morie.fn.hmoob import geron_oob_score

    f = lambda A: np.zeros(len(np.atleast_2d(A)))
    with pytest.raises(ValueError, match="no OOB"):
        geron_oob_score([[0.0], [1.0]], [0, 1], [(f, [True, True])])


def test_hmpas_samples_are_without_replacement():
    from morie.fn.hmpas import geron_pasting

    X = np.arange(8.0).reshape(8, 1)
    y = np.arange(8.0)
    r = geron_pasting(X, y, n_estimators=6, sample_size=3, seed=4)
    for s in r["samples"]:
        assert len(s) == 3 and len(set(s.tolist())) == 3
        assert s.min() >= 0 and s.max() < 8
    const = lambda Xb, yb: (lambda A: np.full(np.atleast_2d(np.asarray(A)).shape[0], 4.0))
    c = geron_pasting(X, y, const, 3, sample_size=2, seed=1)
    assert abs(float(c["train_mse"]) - float(np.mean((4.0 - y) ** 2))) < 1e-12


def test_hmpas_rejects_oversized_samples():
    from morie.fn.hmpas import geron_pasting

    with pytest.raises(ValueError, match="without replacement"):
        geron_pasting([[1.0], [2.0]], [1.0, 2.0], sample_size=5)


def test_hmrsp_feature_sets_and_usage():
    from morie.fn.hmrsp import geron_random_subspaces

    X = np.column_stack([lcg(10, i + 2, -1, 1) for i in range(5)])
    y = lcg(10, 99, 0, 1)
    r = geron_random_subspaces(X, y, n_estimators=12, max_features=2, seed=6)
    assert all(len(s) == 2 and len(set(s.tolist())) == 2 for s in r["feature_sets"])
    assert int(np.sum(r["feature_usage"])) == 12 * 2
    const = lambda Xb, yb: (lambda A: np.full(np.atleast_2d(np.asarray(A)).shape[0], 1.0))
    c = geron_random_subspaces(X, y, const, 4, max_features=3, seed=2)
    assert np.allclose(c["predict"](X), 1.0)


def test_hmrpt_patch_shapes():
    from morie.fn.hmrpt import geron_random_patches

    X = np.column_stack([lcg(9, i + 3, -1, 1) for i in range(4)])
    y = lcg(9, 55, 0, 1)
    r = geron_random_patches(X, y, n_estimators=7, max_samples=4, max_features=2, seed=8)
    for rows, cols in r["patches"]:
        assert len(rows) == 4 and len(set(rows.tolist())) == 4
        assert len(cols) == 2 and len(set(cols.tolist())) == 2
    assert int(np.sum(r["feature_usage"])) == 7 * 2


def test_hmrdt_leaves_are_group_means():
    from morie.fn.hmrdt import geron_regression_tree

    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([1.0, 1.2, 0.8, 8.0, 8.4, 7.6])
    r = geron_regression_tree(X, y, max_depth=1)
    pred = np.asarray(r["predictions"])
    left = pred[:3]
    right = pred[3:]
    assert np.allclose(left, y[:3].mean())
    assert np.allclose(right, y[3:].mean())
    assert abs(float(r["mse"]) - float(np.mean((pred - y) ** 2))) < 1e-12
    assert int(r["n_leaves"]) == 2


def test_hmrdt_depth_limit_is_respected():
    from morie.fn.hmrdt import geron_regression_tree

    X = np.arange(16.0).reshape(16, 1)
    y = np.arange(16.0)
    assert int(geron_regression_tree(X, y, max_depth=2)["depth"]) <= 2
    with pytest.raises(ValueError):
        geron_regression_tree(X, y, max_depth=0)


def test_hmrfc_classifies_a_separable_set():
    from morie.fn.hmrfc import geron_random_forest

    X = np.array([[0.0, 0.0], [0.2, 0.1], [5.0, 5.0], [5.1, 4.9], [0.1, 0.3], [4.8, 5.2]])
    y = np.array([0, 0, 1, 1, 0, 1])
    r = geron_random_forest(X, y, n_estimators=11, seed=3)
    assert float(r["accuracy"]) == 1.0
    assert set(np.asarray(r["predictions"]).tolist()) <= {0.0, 1.0}
    assert int(r["max_features"]) == 2
    assert np.all(np.asarray(r["feature_importance"]) >= 0)


def test_hmrfc_regression_beats_the_constant_predictor():
    from morie.fn.hmrfc import geron_random_forest

    X = np.arange(12.0).reshape(12, 1)
    y = np.where(X.ravel() < 6, 1.0, 9.0)
    r = geron_random_forest(X, y, n_estimators=9, seed=5, task="regression")
    assert float(r["mse"]) < float(np.var(y))


# --------------------------------------------------------------------------
# multiclass and search
# --------------------------------------------------------------------------
def test_hmovo_classifier_count_and_votes():
    from morie.fn.hmovo import geron_one_vs_one

    X = np.array([[0.0], [0.5], [5.0], [5.5], [10.0], [10.5], [15.0], [15.5]])
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    r = geron_one_vs_one(X, y)
    K = 4
    assert int(r["n_classifiers"]) == K * (K - 1) // 2
    assert np.allclose(np.asarray(r["votes"]).sum(axis=1), K * (K - 1) / 2)
    assert float(r["accuracy"]) == 1.0


def test_hmovr_scores_and_positive_rates():
    from morie.fn.hmovr import geron_one_vs_rest

    X = np.array([[0.0], [0.5], [5.0], [5.5], [10.0], [10.5]])
    y = np.array([0, 0, 1, 1, 2, 2])
    r = geron_one_vs_rest(X, y)
    assert int(r["n_classifiers"]) == 3
    assert np.allclose(r["positive_rate"], [2 / 6] * 3)
    assert np.all(np.asarray(r["margin"]) >= 0)
    assert float(r["accuracy"]) == 1.0


def test_hmmto_matches_a_brute_force_leave_one_out_vote():
    from morie.fn.hmmto import geron_multioutput

    X = np.array([[0.0], [1.0], [8.0], [9.0], [20.0]])
    Y = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [2, 2]])
    r = geron_multioutput(X, Y, k=1)
    for i in range(X.shape[0]):
        d = np.abs(X[:, 0] - X[i, 0])
        d[i] = np.inf
        j = int(np.argmin(d))
        assert list(np.asarray(r["predictions"])[i]) == list(Y[j])


def test_hmmto_rejects_k_beyond_the_data():
    from morie.fn.hmmto import geron_multioutput

    with pytest.raises(ValueError):
        geron_multioutput([[0.0], [1.0]], [[0], [1]], k=5)


def test_hmrsc_returns_the_best_of_the_sampled_candidates():
    from morie.fn.hmrsc import geron_randomized_search

    X = np.column_stack([lcg(12, 31, -2, 2)])
    y = 3.0 * X[:, 0]
    r = geron_randomized_search({"alpha": [0.0, 50.0]}, 8, X, y, K=3)
    assert len(r["candidates"]) == 8 == len(r["scores"])
    assert abs(float(r["best_score"]) - max(r["scores"])) < 1e-15
    assert float(r["best_params"]["alpha"]) == 0.0


def test_hmrsc_validates_the_search_space():
    from morie.fn.hmrsc import geron_randomized_search

    X = [[1.0], [2.0], [3.0], [4.0]]
    with pytest.raises(ValueError):
        geron_randomized_search({"alpha": "wide"}, 2, X, [2.0, 4.0, 6.0, 8.0], K=2)
    with pytest.raises(ValueError):
        geron_randomized_search({}, 2, X, [2.0, 4.0, 6.0, 8.0], K=2)


# --------------------------------------------------------------------------
# reinforcement learning
# --------------------------------------------------------------------------
def test_hmpg_gradient_by_hand():
    from morie.fn.hmpg import geron_policy_gradient

    gs = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 2.0])}
    ep = [(0, 0, 1.0), (1, 1, 3.0), (0, 0, 2.0)]
    gamma = 0.5
    G = []
    run = 0.0
    for _, _, rew in reversed(ep):
        run = rew + gamma * run
        G.append(run)
    G = G[::-1]
    want = sum(gs[a] * g for (_, a, _), g in zip(ep, G))
    got = geron_policy_gradient([ep], lambda s, a: gs[a], gamma=gamma)
    assert np.allclose(got["gradient"], want)
    assert np.allclose(got["returns"], G)


def test_hmpg_baseline_centres_the_returns():
    from morie.fn.hmpg import geron_policy_gradient

    ep = [(0, 0, 1.0), (1, 1, 1.0)]
    g = lambda s, a: np.array([1.0, 1.0])
    r = geron_policy_gradient([ep], g, gamma=0.5, baseline=True)
    ret = np.asarray(r["returns"])
    assert abs(float(r["baseline_value"]) - float(ret.mean())) < 1e-12
    assert np.allclose(r["gradient"], np.sum(ret - ret.mean()) * np.array([1.0, 1.0]))


def test_hmrnfc_step_is_eta_times_the_gradient():
    from morie.fn.hmpg import geron_policy_gradient
    from morie.fn.hmrnfc import geron_reinforce

    ep = [[(0, 0, 1.0), (1, 1, 2.0)]]
    g = lambda s, a: (np.array([1.0, 0.0]) if a == 0 else np.array([0.0, 1.0]))
    base = geron_policy_gradient(ep, g, gamma=0.9, baseline=True)
    r = geron_reinforce(ep, g, gamma=0.9, eta=0.25, theta=[1.0, -1.0])
    assert np.allclose(r["step"], 0.25 * np.asarray(base["gradient"]))
    assert np.allclose(r["theta"], np.array([1.0, -1.0]) + np.asarray(r["step"]))


def test_hmrl_return_matches_the_geometric_sum():
    from morie.fn.hmrl import geron_reinforcement_learning

    state = {"t": 0}

    def reset():
        state["t"] = 0
        return 0

    def step(a):
        state["t"] += 1
        return state["t"], 2.0, state["t"] >= 4

    gamma = 0.5
    want = sum(2.0 * gamma**k for k in range(4))
    r = geron_reinforcement_learning({"reset": reset, "step": step}, lambda s: 0, gamma=gamma)
    assert abs(float(r["mean_return"]) - want) < 1e-12
    assert int(r["lengths"][0]) == 4
    assert abs(float(r["effective_horizon"]) - 2.0) < 1e-12


def test_hmrl_reports_truncation():
    from morie.fn.hmrl import geron_reinforcement_learning

    r = geron_reinforcement_learning(
        {"reset": lambda: 0, "step": lambda a: (0, 1.0, False)}, lambda s: 0, gamma=0.9, max_steps=5
    )
    assert int(r["truncated"]) == 1 and int(r["lengths"][0]) == 5


def test_hmppo_learns_the_paying_action_and_respects_the_clip():
    from morie.fn.hmppo import geron_ppo

    r = geron_ppo({"reset": lambda: 0, "step": lambda a: (0, float(a), True)}, [[0.0, 0.0]],
                  epochs=25, lr=0.5, seed=7)
    p = np.asarray(r["probabilities"])[0]
    assert abs(float(p.sum()) - 1.0) < 1e-12
    assert float(p[1]) > 0.8
    assert 0.0 <= float(r["clip_fraction"]) <= 1.0
    assert float(r["return_history"][-1]) >= float(r["return_history"][0])


def test_hmppo_validates_its_knobs():
    from morie.fn.hmppo import geron_ppo

    env = {"reset": lambda: 0, "step": lambda a: (0, 0.0, True)}
    with pytest.raises(ValueError):
        geron_ppo(env, [[0.0, 0.0]], clip_eps=1.5)
    with pytest.raises(ValueError, match="2 actions"):
        geron_ppo(env, [[0.0]])


def test_hmrlhf_converges_to_the_analytic_optimum():
    from morie.fn.hmrlhf import geron_rlhf

    logits = np.array([[0.0, 0.5, -0.25]])
    rew = np.array([[0.0, 1.0, 2.0]])
    beta = 0.5
    ref = np.exp(logits) / np.exp(logits).sum()
    want = ref * np.exp(rew / beta)
    want = want / want.sum()
    r = geron_rlhf(logits, rew, beta=beta, epochs=3000, lr=0.5)
    assert np.allclose(np.asarray(r["policy"])[0], want[0], atol=1e-4)
    assert float(r["objective_history"][-1]) >= float(r["objective_history"][0])


def test_hmrlhf_kl_grows_as_beta_falls():
    from morie.fn.hmrlhf import geron_rlhf

    hot = geron_rlhf([[0.0, 0.0]], [[0.0, 1.0]], beta=0.05, epochs=500)
    cold = geron_rlhf([[0.0, 0.0]], [[0.0, 1.0]], beta=5.0, epochs=500)
    assert float(hot["kl"]) > float(cold["kl"])
    assert float(hot["mean_reward"]) > float(cold["mean_reward"])


# --------------------------------------------------------------------------
# networks
# --------------------------------------------------------------------------
def test_hmregn_predictions_match_a_hand_forward_pass():
    from morie.fn.hmregn import geron_regression_mlp

    X = np.column_stack([lcg(20, 41, -1, 1)])
    y = 2.5 * X[:, 0] + 0.5
    r = geron_regression_mlp(X, y, hidden_sizes=(6,), epochs=300, lr=0.05)
    Ws, bs = r["weights"], r["biases"]
    h = np.maximum(X @ Ws[0] + bs[0], 0.0)
    manual = (h @ Ws[1] + bs[1]).ravel()
    assert np.allclose(np.asarray(r["predictions"]), manual, atol=1e-12)
    assert float(r["loss_history"][-1]) < float(r["loss_history"][0])


def test_hmregn_parameter_count_and_validation():
    from morie.fn.hmregn import geron_regression_mlp

    r = geron_regression_mlp([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0], hidden_sizes=(5, 3), epochs=5)
    want = (2 * 5 + 5) + (5 * 3 + 3) + (3 * 1 + 1)
    assert int(r["n_parameters"]) == want
    with pytest.raises(ValueError):
        geron_regression_mlp([[1.0]], [1.0], epochs=0)


def test_hmrgpt_layer_list_and_parameters():
    from morie.fn.hmrgpt import geron_regression_mlp_pytorch

    r = geron_regression_mlp_pytorch([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0], hidden=(5, 3), epochs=5)
    assert r["layers"] == [
        "Linear(in_features=2, out_features=5)",
        "ReLU()",
        "Linear(in_features=5, out_features=3)",
        "ReLU()",
        "Linear(in_features=3, out_features=1)",
    ]
    assert int(r["n_parameters"]) == (2 * 5 + 5) + (5 * 3 + 3) + (3 * 1 + 1)
    assert bool(r["uses_torch"]) is False


def test_hmprcv_attention_is_softmax_of_scaled_dots():
    from morie.fn.hmprcv import geron_perceiver

    X = np.array([[1.0, 0.5], [-0.5, 2.0], [0.25, 0.25]])
    L0 = np.array([[1.0, -1.0]])
    r = geron_perceiver(X, L0, n_iter=1)
    s = (L0 @ X.T) / math.sqrt(2)
    want = np.exp(s - s.max()) / np.exp(s - s.max()).sum()
    assert np.allclose(r["attention"], want)
    assert np.allclose(r["latents"], L0 + want @ X)
    assert int(r["attention_cost"]) == 3 and int(r["self_attention_cost"]) == 9


def test_hmprcv_checks_widths():
    from morie.fn.hmprcv import geron_perceiver

    with pytest.raises(ValueError):
        geron_perceiver([[1.0, 0.0]], [[1.0, 0.0, 0.0]])


def test_hmprio_decoder_is_a_convex_blend_of_latents():
    from morie.fn.hmprio import geron_perceiver_io

    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    Z0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    Q = np.array([[1.0, 0.0], [0.5, 0.5]])
    r = geron_perceiver_io(X, Z0, Q, n_iter=1)
    A = np.asarray(r["decoder_attention"])
    Z = np.asarray(r["latents"])
    assert np.allclose(A.sum(axis=1), 1.0)
    assert np.allclose(np.asarray(r["outputs"]), A @ Z)
    lo, hi = Z.min(axis=0), Z.max(axis=0)
    assert np.all(np.asarray(r["outputs"]) >= lo - 1e-12) and np.all(np.asarray(r["outputs"]) <= hi + 1e-12)


def test_hmrvat_alpha_matches_the_additive_score():
    from morie.fn.hmrvat import geron_rnn_visual_attention

    F = np.array([[1.0, 0.5], [-1.0, 2.0], [0.25, 0.0]])
    h = np.array([0.3])
    W = np.array([[0.7, -0.2], [0.1, 0.9]])
    U = np.array([[0.4], [-0.3]])
    v = np.array([1.0, -0.5])
    e = np.tanh(F @ W.T + (U @ h)[None, :]) @ v
    want = np.exp(e - e.max()) / np.exp(e - e.max()).sum()
    r = geron_rnn_visual_attention(F, h, W, U, v)
    assert np.allclose(r["alpha"], want)
    assert np.allclose(r["context"], want @ F)
    assert abs(float(np.sum(r["alpha"])) - 1.0) < 1e-12


# --------------------------------------------------------------------------
# convolutional pieces, compression, streaming
# --------------------------------------------------------------------------
def test_hmmxp_matches_a_brute_force_window_scan():
    from morie.fn.hmmxp import geron_max_pool

    x = lcg(36, 77, -5, 5).reshape(6, 6)
    k, s = 3, 2
    out = np.asarray(geron_max_pool(x, k, stride=s)["pooled"])
    oh = (6 - k) // s + 1
    assert out.shape == (oh, oh)
    for i in range(oh):
        for j in range(oh):
            assert abs(out[i, j] - x[i * s : i * s + k, j * s : j * s + k].max()) < 1e-12


def test_hmmxp_pools_channels_independently_and_has_no_parameters():
    from morie.fn.hmmxp import geron_max_pool

    z = np.dstack([np.arange(16.0).reshape(4, 4), np.arange(16.0).reshape(4, 4)[::-1]])
    r = geron_max_pool(z, 2)
    assert r["pooled"].shape == (2, 2, 2)
    assert int(r["parameters"]) == 0
    with pytest.raises(ValueError, match="does not fit"):
        geron_max_pool(np.zeros((2, 2)), 3)


def test_hmpru_prunes_exactly_the_smallest_magnitudes():
    from morie.fn.hmpru import geron_weight_pruning

    w = lcg(20, 101, -3, 3)
    r = geron_weight_pruning(w, 0.35)
    k = int(np.floor(0.35 * 20))
    pruned = np.asarray(r["pruned"])
    assert int(np.sum(pruned == 0.0)) == k
    kept = np.abs(w[pruned != 0.0])
    dropped = np.abs(w[pruned == 0.0])
    assert dropped.max() <= kept.min() + 1e-12
    assert abs(float(r["threshold"]) - float(dropped.max())) < 1e-12


def test_hmpru_schedule_reaches_the_target():
    from morie.fn.hmpru import geron_weight_pruning

    r = geron_weight_pruning([1.0, 2.0, 3.0, 4.0], 0.5, n_rounds=4)
    sched = r["schedule"]
    assert len(sched) == 4
    assert all(sched[i] <= sched[i + 1] + 1e-12 for i in range(3))
    assert abs(sched[-1] - 0.5) < 1e-12


def test_hmptq_round_trip_error_is_bounded_by_half_a_step():
    from morie.fn.hmptq import geron_static_quantization_ptq

    w = lcg(30, 202, -1.5, 1.5)
    cal = lcg(50, 303, -2.0, 3.0)
    r = geron_static_quantization_ptq(w, cal, bits=8)
    assert abs(float(r["weight_scale"]) - float(np.max(np.abs(w))) / 127.0) < 1e-15
    assert abs(float(r["activation_scale"]) - (cal.max() - cal.min()) / 255.0) < 1e-15
    d = np.asarray(r["dequantized_weights"])
    assert np.max(np.abs(d - w)) <= float(r["weight_scale"]) / 2 + 1e-12
    assert np.all(np.abs(np.asarray(r["quantized_weights"])) <= 127)


def test_hmptq_refuses_a_degenerate_calibration_set():
    from morie.fn.hmptq import geron_static_quantization_ptq

    with pytest.raises(ValueError, match="single value"):
        geron_static_quantization_ptq([1.0], [2.0, 2.0])


def test_hmqat_weights_land_on_the_quantization_grid():
    from morie.fn.hmqat import geron_quantization_aware_training

    X = np.column_stack([lcg(15, 404, -2, 2)])
    y = 1.75 * X[:, 0]
    r = geron_quantization_aware_training([0.0], X, y, epochs=400, lr=0.05, bits=6)
    q = np.asarray(r["quantized_weights"])
    scale = float(r["scale"])
    assert np.allclose(q / scale, np.round(q / scale), atol=1e-9)
    assert abs(float(q[0]) - 1.75) < 0.1
    assert float(r["loss_history"][-1]) < float(r["loss_history"][0])


def test_hmqat_coarser_bits_cost_accuracy():
    from morie.fn.hmqat import geron_quantization_aware_training

    X = np.column_stack([lcg(15, 404, -2, 2)])
    y = 1.75 * X[:, 0]
    fine = geron_quantization_aware_training([0.0], X, y, epochs=400, lr=0.05, bits=8)
    coarse = geron_quantization_aware_training([0.0], X, y, epochs=400, lr=0.05, bits=2)
    assert float(coarse["loss"]) >= float(fine["loss"])


def test_hmonl_trajectory_matches_a_hand_sgd_loop():
    from morie.fn.hmonl import geron_online_learning

    X = np.column_stack([lcg(6, 505, -1, 1), np.ones(6)])
    y = lcg(6, 606, -2, 2)
    eta, decay = 0.2, 0.1
    th = np.zeros(2)
    losses = []
    for t in range(6):
        err = float(X[t] @ th - y[t])
        losses.append(err * err)
        th = th - (eta / (1 + decay * t)) * 2 * err * X[t]
    r = geron_online_learning(X, y, eta=eta, decay=decay)
    assert np.allclose(r["theta"], th)
    assert np.allclose(r["losses"], losses)
    assert abs(float(r["cumulative_loss"]) - sum(losses)) < 1e-12


def test_hmonl_rejects_a_bad_rate():
    from morie.fn.hmonl import geron_online_learning

    with pytest.raises(ValueError):
        geron_online_learning([[1.0]], [1.0], eta=0.0)


# --------------------------------------------------------------------------
# language, prompting, generative
# --------------------------------------------------------------------------
def test_hmnsp_input_assembly_and_baseline_logit():
    from morie.fn.hmnsp import geron_next_sentence_prediction

    A, B = ["a", "b", "c"], ["b", "c", "d"]
    r = geron_next_sentence_prediction(A, B, label=1)
    assert r["tokens"] == ["[CLS]"] + A + ["[SEP]"] + B + ["[SEP]"]
    assert list(np.asarray(r["segment_ids"])) == [0] * (len(A) + 2) + [1] * (len(B) + 1)
    overlap = len(set(A) & set(B)) / len(set(A) | set(B))
    logit = 4.0 * overlap - 2.0
    assert abs(float(r["logit"]) - logit) < 1e-12
    p = 1 / (1 + math.exp(-logit))
    assert abs(float(r["loss"]) + math.log(p)) < 1e-12


def test_hmnsp_custom_encoder_needs_its_own_head():
    from morie.fn.hmnsp import geron_next_sentence_prediction

    enc = lambda toks, segs: np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="head weights"):
        geron_next_sentence_prediction(["a"], ["b"], encoder=enc)
    r = geron_next_sentence_prediction(["a"], ["b"], encoder=enc, w=[0.5, 0.25], b=1.0)
    assert abs(float(r["logit"]) - (0.5 * 1.0 + 0.25 * 2.0 + 1.0)) < 1e-12


def test_hmosf_prompt_structure_and_model_contract():
    from morie.fn.hmosf import geron_one_shot

    seen = {}

    def model(prompt):
        seen["prompt"] = prompt
        return "ok"

    r = geron_one_shot(model, ("x1", "y1"), "xq")
    assert seen["prompt"] == [("x1", "y1"), ("xq", None)]
    assert r["prediction"] == "ok" and int(r["shots"]) == 1
    with pytest.raises(ValueError):
        geron_one_shot(None, ("x1", "y1"), "xq")
    with pytest.raises(ValueError):
        geron_one_shot(lambda p: None, ("x1", "y1"), "xq")


def test_hmnmt_loss_is_the_hand_log_sum():
    from morie.fn.hmnmt import geron_encoder_decoder_nmt

    table = {0: [0.6, 0.3, 0.1], 1: [0.1, 0.7, 0.2], 2: [0.2, 0.2, 0.6]}
    model = {"encode": lambda s: 0, "decode": lambda z, prefix: table[len(prefix)]}
    tgt = [2, 0, 1]
    want = -sum(math.log(table[i][tgt[i]]) for i in range(3))
    r = geron_encoder_decoder_nmt([9, 9], tgt, model)
    assert abs(float(r["loss"]) - want) < 1e-12
    assert abs(float(r["perplexity"]) - math.exp(want / 3)) < 1e-12
    assert [int(t) for t in r["greedy"]] == [0, 1, 2]


def test_hmnmt_rejects_a_non_distribution():
    from morie.fn.hmnmt import geron_encoder_decoder_nmt

    model = {"encode": lambda s: 0, "decode": lambda z, p: [0.5, 0.9]}
    with pytest.raises(ValueError, match="probability vector"):
        geron_encoder_decoder_nmt([1], [0], model)


def test_hmncsn_recovers_the_exact_gaussian_score():
    from morie.fn.hmncsn import geron_ncsn

    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    var = float(((X - X.mean()) ** 2).mean())
    r = geron_ncsn(X, [1.0], epochs=800)
    assert abs(float(r["analytic"][0]["W"][0, 0]) + 1.0 / (var + 1.0)) < 1e-12
    assert abs(float(r["models"][0]["W"][0, 0]) + 1.0 / (var + 1.0)) < 0.05
    assert float(r["loss_history"][0][-1]) < float(r["loss_history"][0][0])


def test_hmncsn_ladder_is_ordered_and_validated():
    from morie.fn.hmncsn import geron_ncsn

    X = np.array([[-1.0], [0.0], [1.0]])
    r = geron_ncsn(X, [0.5, 2.0], epochs=100)
    assert [m["sigma"] for m in r["models"]] == [2.0, 0.5]
    with pytest.raises(ValueError):
        geron_ncsn(X, [-1.0])


# --------------------------------------------------------------------------
# systems, deployment, architecture
# --------------------------------------------------------------------------
def test_hmmpp_partition_is_optimal_by_exhaustive_search():
    from itertools import combinations

    from morie.fn.hmmpp import geron_model_parallelism

    w = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]
    k = 3
    best = min(
        max(sum(w[a:b]) for a, b in zip((0,) + cuts, cuts + (len(w),)))
        for cuts in combinations(range(1, len(w)), k - 1)
    )
    r = geron_model_parallelism(w, k)
    assert abs(float(r["max_load"]) - best) < 1e-9
    assert len(set(np.asarray(r["assignment"]).tolist())) == k
    assert abs(float(np.sum(r["device_loads"])) - sum(w)) < 1e-9


def test_hmmpp_refuses_more_devices_than_layers():
    from morie.fn.hmmpp import geron_model_parallelism

    with pytest.raises(ValueError):
        geron_model_parallelism([1, 2], 3)


def test_hmppp_bubble_formula_and_schedule():
    from morie.fn.hmppp import geron_pipeline_parallelism

    S, M = 3, 5
    r = geron_pipeline_parallelism([1] * 6, S, n_microbatches=M)
    assert abs(float(r["bubble_fraction"]) - (S - 1) / (M + S - 1)) < 1e-12
    assert int(r["n_slots"]) == M + S - 1
    sched = np.asarray(r["schedule"])
    assert sched.shape == (S, M + S - 1)
    for s in range(S):
        assert [v for v in sched[s] if v >= 0] == list(range(M))
    assert abs(float(r["utilisation"]) + float(r["bubble_fraction"]) - 1.0) < 1e-12


def test_hmmxp2_underflow_and_scale_recommendation():
    from morie.fn.hmmxp2 import geron_mixed_precision

    g = [1e-8, 1e-3, 0.5]
    scale = 256.0
    want_under = sum(1 for v in g if 0 < abs(v) * scale < 6.103515625e-05)
    r = geron_mixed_precision([1.0, 2.0, 3.0], loss_scale=scale, grads=g)
    assert int(r["n_underflow"]) == want_under
    assert abs(float(r["max_safe_loss_scale"]) - 65504.0 / 0.5) < 1e-9
    rec = float(r["recommended_loss_scale"])
    assert rec <= float(r["max_safe_loss_scale"])
    assert abs(math.log2(rec) - round(math.log2(rec))) < 1e-12
    assert int(r["memory_bytes_fp16"]) * 2 == int(r["memory_bytes_fp32"])


def test_hmmxp2_flags_overflow():
    from morie.fn.hmmxp2 import geron_mixed_precision

    assert bool(geron_mixed_precision([1.0], loss_scale=1e5, grads=[1.0])["overflow"]) is True
    assert bool(geron_mixed_precision([1.0], loss_scale=1.0, grads=[1.0])["overflow"]) is False


def test_hmonnx_traces_shapes_and_parameters():
    from morie.fn.hmonnx import geron_onnx_export

    m = [
        {"op": "Flatten"},
        {"op": "Gemm", "in_features": 12, "out_features": 5},
        {"op": "Relu"},
        {"op": "Gemm", "in_features": 5, "out_features": 2},
    ]
    r = geron_onnx_export(m, np.zeros((4, 3, 4)))
    assert r["input_shape"] == (4, 3, 4) and r["output_shape"] == (4, 2)
    assert int(r["n_parameters"]) == 12 * 5 + 5 + 5 * 2 + 2
    assert [n["op"] for n in r["nodes"]] == ["Flatten", "Gemm", "Relu", "Gemm"]
    assert bool(r["is_protobuf"]) is False


def test_hmonnx_refuses_unknown_ops_and_bad_shapes():
    from morie.fn.hmonnx import geron_onnx_export

    with pytest.raises(ValueError, match="unsupported op"):
        geron_onnx_export([{"op": "Wormhole"}], np.zeros((1, 2)))
    with pytest.raises(ValueError, match="input features"):
        geron_onnx_export([{"op": "Gemm", "in_features": 4, "out_features": 1}], np.zeros((1, 2)))


def test_hmpttn_reports_the_real_buffer_geometry():
    from morie.fn.hmpttn import geron_pytorch_tensor

    r = geron_pytorch_tensor(np.arange(6).reshape(2, 3), dtype="int16")
    t = np.asarray(r["tensor"])
    assert r["shape"] == (2, 3) and r["dtype"] == "int16"
    assert int(r["nbytes"]) == t.size * t.itemsize == 12
    assert bool(geron_pytorch_tensor([[1.0]])["dtype_changed"]) is True
    with pytest.raises(ValueError):
        geron_pytorch_tensor([1.0], dtype="float3")
    with pytest.raises(ValueError):
        geron_pytorch_tensor([1.0], device="tpu")


def test_hmpmps_demotion_error_matches_a_hand_cast():
    from morie.fn.hmpmps import geron_mps_acceleration

    x = lcg(12, 808, -1, 1)
    r = geron_mps_acceleration(x)
    want = float(np.max(np.abs(x.astype(np.float32).astype(np.float64) - x)))
    assert r["dtype_on_device"] == "float32"
    assert abs(float(r["max_abs_error"]) - want) < 1e-18
    assert bool(r["executes_on_metal"]) is False
    with pytest.raises(ValueError, match="complex"):
        geron_mps_acceleration(np.array([1 + 2j]))


def test_hmpvt_patch_embedding_equals_the_patch_mean():
    from morie.fn.hmpvt import geron_pvt

    img = lcg(6 * 6 * 2, 909, -1, 1).reshape(6, 6, 2)
    W = np.ones((3 * 3 * 2, 1)) / 18.0
    r = geron_pvt(img, [{"patch_size": 3, "dim": 1, "W": W}])
    tok = np.asarray(r["tokens"])
    assert tok.shape == (2, 2, 1)
    for i in range(2):
        for j in range(2):
            patch = img[i * 3 : i * 3 + 3, j * 3 : j * 3 + 3, :]
            assert abs(float(tok[i, j, 0]) - float(patch.mean())) < 1e-12


def test_hmpvt_costs_and_shape_validation():
    from morie.fn.hmpvt import geron_pvt

    r = geron_pvt(np.zeros((8, 8, 1)), [{"patch_size": 2, "dim": 4, "sr_ratio": 2}])
    n = 16
    assert int(r["full_attention_cost"]) == n * n
    assert int(r["attention_cost"]) == n * (n // 4)
    with pytest.raises(ValueError, match="does not divide"):
        geron_pvt(np.zeros((5, 5, 1)), [{"patch_size": 2, "dim": 2}])


def test_hmpemb_coverage_and_freezing():
    from morie.fn.hmpemb import geron_pretrained_embeddings

    pre = {"a": [1.0, 2.0], "b": [3.0, 4.0]}
    r = geron_pretrained_embeddings(["a", "q", "b", "z"], pre, freeze=True)
    E = np.asarray(r["embeddings"])
    assert np.allclose(E[0], pre["a"]) and np.allclose(E[2], pre["b"])
    assert abs(float(r["coverage"]) - 0.5) < 1e-12
    assert r["oov"] == ["q", "z"] and r["oov_indices"] == [1, 3]
    assert int(r["trainable"]) == 0 and int(r["n_parameters"]) == 8
    assert np.abs(E[[1, 3]]).max() <= 0.05
    with pytest.raises(ValueError, match="mixed widths"):
        geron_pretrained_embeddings(["a"], {"a": [1.0], "b": [1.0, 2.0]})


# --------------------------------------------------------------------------
# a mean-of-inputs stub would pass none of these
# --------------------------------------------------------------------------
def test_no_module_returns_the_mean_of_its_inputs():
    from morie.fn.hmmsec import geron_linreg_mse_cost
    from morie.fn.hmneq import geron_normal_equation
    from morie.fn.hmpre import geron_precision
    from morie.fn.hmrelu import geron_relu
    from morie.fn.hmrms import geron_rmse

    y = [1.0, 5.0, 9.0]
    mean = float(np.mean(y))
    assert abs(float(geron_rmse([0.0, 0.0, 0.0], [6.0, 8.0, 0.0])["estimate"]) - float(np.mean([0, 0, 0, 6, 8, 0]))) > 1.0
    assert abs(float(geron_precision([1, 0, 0, 0], [1, 1, 0, 0])["estimate"]) - float(np.mean([1, 0, 0, 0]))) > 0.2
    assert np.max(np.abs(np.asarray(geron_relu(y)["estimate"]) - mean)) > 1.0
    X = np.column_stack([np.ones(3), [0.0, 1.0, 2.0]])
    assert np.max(np.abs(np.asarray(geron_normal_equation(X, y)["estimate"]) - mean)) > 1.0
    assert abs(float(geron_linreg_mse_cost(X, y, [0.0, 0.0])["estimate"]) - mean) > 1.0
