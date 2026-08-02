# morie.fn -- test file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent-route checks for the w4b tranche of morie.fn (Géron).

Every assertion here is derived by a route the implementation does not
take: central finite differences against analytic gradients, brute-force
counting against vectorised metrics, hand log-sums against losses,
closed-form solutions against iterative fits, and invariants (bounds,
monotonicity, partition, conservation) that a mean-of-inputs stub cannot
satisfy.
"""

import math

from morie.fn import _array_core as np
import pytest

# ── deterministic data --------------------------------------------------


def lcg(n, seed=12345):
    """The house LCG. Returns n uniforms in (0, 1)."""
    s = seed
    out = []
    for _ in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out.append((s + 0.5) / 2**32)
    return np.asarray(out)


def lcg_matrix(rows, cols, seed=999):
    return lcg(rows * cols, seed).reshape(rows, cols)


def central_diff(f, x, i, h=1e-6):
    """d f / d x_i by central difference."""
    up = np.array(x, dtype=float)
    dn = np.array(x, dtype=float)
    up[i] += h
    dn[i] -= h
    return (f(up) - f(dn)) / (2 * h)


# ── hmgpt3 / hmmis7 : architecture accounting ---------------------------


def test_gpt3_parameter_count_matches_hand_arithmetic():
    from morie.fn.hmgpt3 import geron_gpt3

    d, L, V, ctx = 12288, 96, 50257, 2048
    per_layer = 12 * d * d + 13 * d  # 4d^2 attn + 8d^2 mlp + 13d biases/norms
    expected = L * per_layer + V * d + ctx * d + 2 * d
    r = geron_gpt3([1, 2, 3], n_tokens=4)
    assert r["parameters_per_layer"] == per_layer
    assert r["total_parameters"] == expected
    assert r["total_parameters"] == sum(r["breakdown"].values())


def test_gpt3_kv_cache_is_linear_in_sequence_length():
    from morie.fn.hmgpt3 import geron_gpt3

    a = geron_gpt3([1] * 10, n_tokens=0)["kv_cache_bytes"]
    b = geron_gpt3([1] * 20, n_tokens=0)["kv_cache_bytes"]
    assert b == 2 * a


def test_gpt3_rejects_overlong_context():
    from morie.fn.hmgpt3 import geron_gpt3

    with pytest.raises(ValueError, match="context window"):
        geron_gpt3([1] * 2048, n_tokens=1)


def test_mistral_gqa_cache_saving_equals_head_ratio():
    from morie.fn.hmmis7 import geron_mistral7b

    r = geron_mistral7b([1, 2, 3], n_tokens=1)
    assert r["kv_cache_saving"] == pytest.approx(32 / 8)
    # parameters counted independently
    d, L, V, dff = 4096, 32, 32000, 14336
    per = 2 * d * d + 2 * d * 1024 + 3 * d * dff + 2 * d
    assert r["parameters_per_layer"] == per
    assert r["total_parameters"] == L * per + 2 * V * d + d


def test_mistral_sliding_window_mask_counts_by_hand():
    from morie.fn.hmmis7 import geron_mistral7b

    m = geron_mistral7b(list(range(10)), n_tokens=0, window=3)["attention_mask"]
    for i in range(10):
        expect = min(i + 1, 3)
        assert int(m[i].sum()) == expect
    assert not m[0, 1]  # strictly causal


# ── hmgrp / hmjl : random projection ------------------------------------


def test_gaussian_projection_is_exactly_X_times_R():
    from morie.fn.hmgrp import geron_gaussian_rand_projection

    X = lcg_matrix(6, 4, seed=1)
    r = geron_gaussian_rand_projection(X, d_out=3, seed=7)
    assert np.allclose(r["X_projected"], X @ r["R"])
    assert r["R"].shape == (4, 3)


def test_gaussian_projection_preserves_norms_in_expectation():
    from morie.fn.hmgrp import geron_gaussian_rand_projection

    # E||R^T u||^2 = ||u||^2 : average the ratio over many seeds.
    u = np.array([[1.0, -2.0, 0.5, 3.0]])
    ratios = []
    for s in range(200):
        z = geron_gaussian_rand_projection(u, d_out=8, seed=s)["X_projected"]
        ratios.append(float(np.sum(z**2)) / float(np.sum(u**2)))
    assert abs(np.mean(ratios) - 1.0) < 0.05


def test_jl_bound_matches_formula_and_is_dimension_free():
    from morie.fn.hmjl import geron_johnson_lindenstrauss

    n, eps = 500, 0.3
    hand = 4.0 * math.log(n) / (eps**2 / 2 - eps**3 / 3)
    r = geron_johnson_lindenstrauss(n, eps)
    assert r["d_min_exact"] == pytest.approx(hand)
    assert r["d_min"] == math.ceil(hand)
    # monotone in eps
    assert geron_johnson_lindenstrauss(n, 0.1)["d_min"] > geron_johnson_lindenstrauss(n, 0.4)["d_min"]


def test_jl_rejects_eps_outside_unit_interval():
    from morie.fn.hmjl import geron_johnson_lindenstrauss

    for bad in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(ValueError):
            geron_johnson_lindenstrauss(10, bad)


# ── hmgru / hmlstm / hmmcel : recurrent cells ---------------------------


def _zero_weights(keys, n_units, n_in):
    W = {}
    for k in keys:
        if k.startswith("b"):
            W[k] = np.zeros(n_units)
        elif k.startswith("W"):
            W[k] = np.zeros((n_units, n_in))
        else:
            W[k] = np.zeros((n_units, n_units))
    return W


def test_gru_matches_hand_computed_step():
    from morie.fn.hmgru import geron_gru

    keys = ("W_z", "U_z", "b_z", "W_r", "U_r", "b_r", "W_h", "U_h", "b_h")
    W = _zero_weights(keys, 1, 1)
    W["W_z"] = np.array([[1.0]])
    W["W_h"] = np.array([[2.0]])
    x, h = [1.0], [0.5]
    z = 1 / (1 + math.exp(-1.0))
    r_gate = 0.5
    h_tilde = math.tanh(2.0)
    expect = (1 - z) * 0.5 + z * h_tilde
    out = geron_gru(x, h, W)
    assert float(out["z_t"][0]) == pytest.approx(z)
    assert float(out["r_t"][0]) == pytest.approx(r_gate)
    assert float(out["h_t"][0]) == pytest.approx(expect)


def test_gru_zero_update_gate_copies_state():
    from morie.fn.hmgru import geron_gru

    keys = ("W_z", "U_z", "b_z", "W_r", "U_r", "b_r", "W_h", "U_h", "b_h")
    W = _zero_weights(keys, 2, 2)
    W["b_z"] = np.array([-50.0, -50.0])
    out = geron_gru([1.0, 1.0], [7.0, -3.0], W)
    assert np.allclose(out["h_t"], [7.0, -3.0])


def test_lstm_cell_state_is_gated_sum():
    from morie.fn.hmlstm import geron_lstm

    keys = ("W_i", "U_i", "b_i", "W_f", "U_f", "b_f", "W_o", "U_o", "b_o", "W_g", "U_g", "b_g")
    W = _zero_weights(keys, 1, 1)
    W["b_f"] = np.array([1.0])
    W["b_i"] = np.array([-1.0])
    W["b_g"] = np.array([0.5])
    f = 1 / (1 + math.exp(-1.0))
    i = 1 / (1 + math.exp(1.0))
    g = math.tanh(0.5)
    o = 0.5
    c_prev = 2.0
    c = f * c_prev + i * g
    out = geron_lstm([0.0], [0.0], [c_prev], W)
    assert float(out["c_t"][0]) == pytest.approx(c)
    assert float(out["h_t"][0]) == pytest.approx(o * math.tanh(c))


def test_memory_cell_unrolls_a_geometric_series():
    from morie.fn.hmmcel import geron_memory_cell

    leaky = lambda c, x: 0.5 * np.asarray(c) + np.asarray(x)
    r = geron_memory_cell([0.0], [[1.0]] * 6, leaky)
    hand = 0.0
    for _ in range(6):
        hand = 0.5 * hand + 1.0
    assert float(r["c_t"][0]) == pytest.approx(hand)
    assert r["n_steps"] == 6


def test_memory_cell_enforces_shape_contract():
    from morie.fn.hmmcel import geron_memory_cell

    with pytest.raises(ValueError, match="shape"):
        geron_memory_cell([0.0, 0.0], [1.0, 1.0], lambda c, x: np.zeros(3))


# ── hmhebb / hmhei / hmhev : layer primitives ---------------------------


def test_hebb_rule_equals_brute_force_outer_products():
    from morie.fn.hmhebb import geron_hebb_rule

    X = lcg_matrix(5, 3, seed=11)
    Y = lcg_matrix(5, 2, seed=22)
    eta = 0.3
    hand = np.zeros((3, 2))
    for i in range(5):
        for a in range(3):
            for b in range(2):
                hand[a, b] += eta * X[i, a] * Y[i, b]
    assert np.allclose(geron_hebb_rule(X, Y, eta=eta)["dW"], hand)


def test_he_init_variance_matches_2_over_fan_in():
    from morie.fn.hmhei import geron_he_init

    r = geron_he_init(100, seed=4, fan_out=200)
    assert r["var_target"] == pytest.approx(2 / 100)
    assert r["std_target"] == pytest.approx(math.sqrt(2 / 100))
    # empirical sd of 20000 draws is close to target
    assert abs(r["empirical_std"] - r["std_target"]) < 0.05 * r["std_target"]


def test_heaviside_counts_active_units_by_brute_force():
    from morie.fn.hmhev import geron_heaviside

    z = lcg(50, seed=31) - 0.5
    r = geron_heaviside(z)
    assert r["n_active"] == sum(1 for v in z if v >= 0)
    assert np.all((r["activation"] == 0.0) | (r["activation"] == 1.0))
    assert np.all(r["derivative"] == 0.0)


# ── activations: hmlrel, hmmish ----------------------------------------


def test_leaky_relu_derivative_matches_finite_difference():
    from morie.fn.hmlrel import geron_leaky_relu

    for z0 in (-3.0, -0.4, 0.7, 5.0):
        f = lambda v: float(geron_leaky_relu(v, alpha=0.2)["activation"][0])
        num = central_diff(f, [z0], 0)
        ana = float(geron_leaky_relu([z0], alpha=0.2)["derivative"][0])
        assert num == pytest.approx(ana, abs=1e-6)


def test_mish_derivative_matches_finite_difference_and_is_nonmonotonic():
    from morie.fn.hmmish import geron_mish

    f = lambda v: float(geron_mish(v)["activation"][0])
    for z0 in (-2.5, -1.0, 0.3, 4.0):
        num = central_diff(f, [z0], 0)
        ana = float(geron_mish([z0])["derivative"][0])
        assert num == pytest.approx(ana, abs=1e-6)
    grid = np.linspace(-5, 5, 401)
    vals = geron_mish(grid)["activation"]
    assert np.min(vals) < 0.0  # dips below zero: not monotone
    assert np.all(np.isfinite(geron_mish([700.0, -700.0])["activation"]))


# ── schedules: hmlcos, hmlrex, hmlrs, hml1c ----------------------------


def test_cosine_annealing_endpoints_and_monotonicity():
    from morie.fn.hmlcos import geron_cosine_annealing

    T, hi, lo = 20, 0.4, 0.05
    sch = geron_cosine_annealing(0, T=T, eta_max=hi, eta_min=lo)["schedule"]
    assert sch[0] == pytest.approx(hi)
    assert sch[-1] == pytest.approx(lo)
    assert sch[T // 2] == pytest.approx((hi + lo) / 2)
    assert np.all(np.diff(sch) < 0)


def test_exponential_decay_matches_geometric_series():
    from morie.fn.hmlrex import geron_lr_exponential

    r = geron_lr_exponential(0.2, 0.9, [0, 1, 2, 10])
    assert np.allclose(r["eta"], [0.2 * 0.9**t for t in (0, 1, 2, 10)])
    # steps to fall tenfold, verified by evaluating the schedule there
    spd = r["steps_per_decade"]
    at = float(geron_lr_exponential(0.2, 0.9, spd)["eta"])
    assert at == pytest.approx(0.02)


def test_power_schedule_satisfies_robbins_monro_partial_sums():
    from morie.fn.hmlrs import geron_learning_rate_schedule

    r = geron_learning_rate_schedule(500, eta0=1.0, t0=1.0)
    hand_sum = sum(1.0 / (t + 1.0) for t in range(501))
    hand_sq = sum((1.0 / (t + 1.0)) ** 2 for t in range(501))
    assert r["sum_eta"] == pytest.approx(hand_sum)
    assert r["sum_eta_squared"] == pytest.approx(hand_sq)
    assert r["sum_eta_squared"] < math.pi**2 / 6  # bounded, unlike sum_eta


def test_one_cycle_peaks_in_the_middle_with_mirrored_momentum():
    from morie.fn.hml1c import geron_one_cycle

    T = 11
    lrs = [float(geron_one_cycle(t, T, 0.5, 0.05)["lr"]) for t in range(T)]
    moms = [float(geron_one_cycle(t, T, 0.5, 0.05)["momentum"]) for t in range(T)]
    peak = int(np.argmax(lrs))
    assert lrs[peak] == pytest.approx(0.5)
    assert moms[peak] == pytest.approx(min(moms))
    assert np.all(np.diff(lrs[: peak + 1]) > 0)
    assert np.all(np.diff(lrs[peak:]) < 0)


# ── regularisation: hml1r, hml2r, hmmnr, hmlaso ------------------------


def test_l1_penalty_and_prox_by_hand():
    from morie.fn.hml1r import geron_l1_regularization

    theta = [2.0, -0.3, 0.0, 5.5]
    a = 0.4
    r = geron_l1_regularization(theta, a)
    assert r["penalty"] == pytest.approx(a * sum(abs(v) for v in theta))
    hand_prox = [math.copysign(max(abs(v) - a, 0.0), v) for v in theta]
    assert np.allclose(r["prox"], hand_prox)
    assert r["n_zero"] == sum(1 for v in hand_prox if v == 0)


def test_l2_gradient_matches_finite_difference_of_penalty():
    from morie.fn.hml2r import geron_l2_regularization

    theta = np.array([1.5, -2.0, 0.25])
    f = lambda v: geron_l2_regularization(v, 0.7)["penalty"]
    g = geron_l2_regularization(theta, 0.7)["gradient"]
    for i in range(3):
        assert central_diff(f, theta, i) == pytest.approx(g[i], abs=1e-6)


def test_max_norm_projects_onto_the_ball_preserving_direction():
    from morie.fn.hmmnr import geron_max_norm

    w = np.array([3.0, -4.0, 12.0])
    r = geron_max_norm(w, r=2.0)
    assert np.linalg.norm(r["w"]) == pytest.approx(2.0)
    cos = float(r["w"] @ w / (np.linalg.norm(r["w"]) * np.linalg.norm(w)))
    assert cos == pytest.approx(1.0)
    inside = geron_max_norm([0.1, 0.2], r=1.0)
    assert np.allclose(inside["w"], [0.1, 0.2]) and not inside["clipped"]


def test_lasso_gradient_matches_finite_difference_away_from_kink():
    from morie.fn.hmlaso import geron_lasso_cost

    X = lcg_matrix(8, 3, seed=77)
    y = lcg(8, seed=88) * 4
    theta = np.array([0.6, -1.3, 2.2])
    f = lambda v: geron_lasso_cost(X, y, v, alpha=0.35)["cost"]
    g = geron_lasso_cost(X, y, theta, alpha=0.35)["gradient"]
    for i in range(3):
        assert central_diff(f, theta, i) == pytest.approx(g[i], abs=1e-5)


# ── logistic family: hmlogp, hmlogcl, hmlogg ---------------------------


def test_logistic_probability_against_hand_sigmoid():
    from morie.fn.hmlogp import geron_logistic_probability

    X = [[1.0, 2.0], [1.0, -1.0]]
    th = [0.5, -0.25]
    hand = [1 / (1 + math.exp(-(0.5 - 0.5))), 1 / (1 + math.exp(-(0.5 + 0.25)))]
    assert np.allclose(geron_logistic_probability(X, th)["p_hat"], hand)


def test_logistic_cost_equals_hand_log_sum():
    from morie.fn.hmlogcl import geron_logistic_cost

    X = [[1.0], [1.0], [1.0]]
    th = [0.8]
    y = [1, 0, 1]
    p = 1 / (1 + math.exp(-0.8))
    hand = -(math.log(p) + math.log(1 - p) + math.log(p)) / 3
    assert geron_logistic_cost(X, y, th)["cost"] == pytest.approx(hand)


def test_logistic_gradient_matches_finite_difference_of_the_cost():
    from morie.fn.hmlogcl import geron_logistic_cost
    from morie.fn.hmlogg import geron_logistic_gradient

    X = np.hstack([np.ones((10, 1)), lcg_matrix(10, 2, seed=5) * 3 - 1.5])
    y = (lcg(10, seed=6) > 0.5).astype(int)
    theta = np.array([0.2, -0.9, 1.1])
    f = lambda v: geron_logistic_cost(X, y, v)["cost"]
    g = geron_logistic_gradient(X, y, theta)["gradient"]
    for i in range(3):
        assert central_diff(f, theta, i) == pytest.approx(g[i], abs=1e-6)
    H = geron_logistic_gradient(X, y, theta)["hessian"]
    assert np.all(np.linalg.eigvalsh(H) >= -1e-12)


def test_logistic_cost_rejects_non_binary_labels():
    from morie.fn.hmlogcl import geron_logistic_cost

    with pytest.raises(ValueError, match="only 0 and 1"):
        geron_logistic_cost([[1.0], [1.0]], [0, 2], [0.0])


# ── optimizers: hmmom, hmmbgd, hmlrpt ----------------------------------


def test_momentum_first_two_steps_by_hand():
    from morie.fn.hmmom import geron_momentum

    s1 = geron_momentum([2.0], v=[0.0], beta=0.8, eta=0.1)
    assert float(s1["v"][0]) == pytest.approx(2.0)
    assert float(s1["step"][0]) == pytest.approx(-0.2)
    s2 = geron_momentum([2.0], v=s1["v"], beta=0.8, eta=0.1)
    assert float(s2["v"][0]) == pytest.approx(0.8 * 2.0 + 2.0)
    assert float(s2["step"][0]) == pytest.approx(-0.1 * 3.6)
    # terminal velocity g/(1-beta)
    v = 0.0
    for _ in range(500):
        v = 0.8 * v + 2.0
    assert v == pytest.approx(2.0 / 0.2)
    assert float(s1["terminal_step"][0]) == pytest.approx(0.1 * 10.0)


def test_minibatch_gd_gradient_matches_finite_difference_of_mse():
    from morie.fn.hmmbgd import geron_minibatch_gd

    X = np.hstack([np.ones((6, 1)), lcg_matrix(6, 1, seed=41) * 5])
    y = lcg(6, seed=42) * 10
    theta = np.array([0.3, 1.7])
    mse = lambda v: float(np.mean((X @ v - y) ** 2))
    g = geron_minibatch_gd(X, y, theta, eta=1e-8, b=6)["gradient"]
    for i in range(2):
        assert central_diff(mse, theta, i) == pytest.approx(g[i], abs=1e-5)


def test_sgd_linear_regression_reaches_the_normal_equations():
    from morie.fn.hmlrpt import geron_linreg_pytorch

    X = lcg_matrix(30, 2, seed=101) * 4 - 2
    true = np.array([1.5, -0.75])
    y = X @ true + 0.3
    r = geron_linreg_pytorch(X, y, epochs=5000, lr=0.05)
    Xb = np.hstack([np.ones((30, 1)), X])
    cf = np.linalg.lstsq(Xb, y, rcond=None)[0]
    assert np.allclose(np.concatenate([[r["b"]], r["w"]]), cf, atol=1e-6)
    assert np.allclose(r["w"], true, atol=1e-6)


# ── metrics: hmmae, hmmms, hmimp ---------------------------------------


def test_mae_equals_brute_force_mean_absolute_residual():
    from morie.fn.hmmae import geron_mae

    yt = lcg(40, seed=201) * 10
    yp = lcg(40, seed=202) * 10
    hand = sum(abs(a - b) for a, b in zip(yt, yp)) / 40
    r = geron_mae(yt, yp)
    assert r["mae"] == pytest.approx(hand)
    assert r["mae"] <= r["rmse"] + 1e-12  # l1/m never exceeds l2/sqrt(m)
    assert geron_mae([1.0, 2.0], [1.0, 2.0])["mae"] == 0.0


def test_mae_is_not_a_mean_of_inputs():
    from morie.fn.hmmae import geron_mae

    # residuals are 1, 1, 1, so the MAE is exactly 1; a stub returning
    # mean(y_true) would give 5.0
    r = geron_mae([5.0, 5.0, 5.0], [4.0, 6.0, 4.0])
    assert r["mae"] == pytest.approx(1.0)
    assert r["mae"] != pytest.approx(5.0)
    assert r["estimate"] != pytest.approx(float(np.mean([5.0, 5.0, 5.0])))


def test_min_max_scaling_hits_the_endpoints_per_column():
    from morie.fn.hmmms import geron_min_max_scaling

    X = lcg_matrix(20, 3, seed=303) * np.array([1.0, 100.0, 0.01])
    Z = geron_min_max_scaling(X)["X_scaled"]
    assert np.allclose(Z.min(axis=0), 0.0)
    assert np.allclose(Z.max(axis=0), 1.0)
    with pytest.raises(ValueError, match="constant"):
        geron_min_max_scaling([[1.0], [1.0], [1.0]])


def test_median_imputation_uses_the_median_not_the_mean():
    from morie.fn.hmimp import geron_imputation_median

    col = [1.0, 2.0, 3.0, 1000.0, np.nan]
    r = geron_imputation_median(np.asarray(col).reshape(-1, 1))
    present = [1.0, 2.0, 3.0, 1000.0]
    assert r["statistics"][0] == pytest.approx(float(np.median(present)))
    assert r["statistics"][0] != pytest.approx(float(np.mean(present)))
    assert r["n_missing"] == 1
    with pytest.raises(ValueError, match="entirely missing"):
        geron_imputation_median([[np.nan], [np.nan]])


# ── trees / boosting: hmigr, hmhgb -------------------------------------


def test_information_gain_matches_hand_entropy():
    from morie.fn.hmigr import geron_information_gain

    y = [0, 0, 0, 1, 1, 1, 1, 1]

    def H(labels):
        vals, counts = np.unique(labels, return_counts=True)
        p = counts / len(labels)
        return float(-np.sum(p * np.log2(p)))

    split = [True, True, True, False, False, False, False, False]
    hand = H(y) - (3 / 8) * H([0, 0, 0]) - (5 / 8) * H([1] * 5)
    r = geron_information_gain(y, split)
    assert r["information_gain"] == pytest.approx(hand)
    assert r["parent_entropy"] == pytest.approx(H(y))


def test_information_gain_is_never_negative_over_random_splits():
    from morie.fn.hmigr import geron_information_gain

    y = (lcg(60, seed=606) > 0.4).astype(int)
    for s in range(20):
        split = lcg(60, seed=700 + s) > 0.5
        assert geron_information_gain(y, split)["information_gain"] >= -1e-12


def test_histogram_boosting_lowers_the_mse_every_round():
    from morie.fn.hmhgb import geron_histogram_gradient_boosting

    X = lcg_matrix(60, 2, seed=808) * 4
    y = 3.0 * (X[:, 0] > 2.0) + 1.5 * (X[:, 1] > 3.0)
    r = geron_histogram_gradient_boosting(X, y, max_iter=60, learning_rate=0.2)
    hist = r["mse_history"]
    assert np.all(np.diff(hist) <= 1e-12)
    assert hist[0] == pytest.approx(float(np.var(y)))  # baseline is the mean
    assert hist[-1] < 0.05 * hist[0]


def test_histogram_boosting_binning_is_the_only_approximation():
    from morie.fn.hmhgb import geron_histogram_gradient_boosting

    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = np.where(X.ravel() < 5.5, 0.0, 4.0)
    fine = geron_histogram_gradient_boosting(X, y, max_iter=60, learning_rate=0.3)
    coarse = geron_histogram_gradient_boosting(X, y, max_iter=60, learning_rate=0.3, max_bins=2)
    assert fine["train_mse"] < 1e-8
    assert coarse["bins_used"] == [2]
    assert coarse["train_mse"] >= fine["train_mse"]


# ── clustering: hmkmpp, hmkmn, hmmbkm, hmiseg, hmkmlim, hmmnsh ---------


def test_kmeans_inertia_equals_brute_force_within_cluster_sums():
    from morie.fn.hmkmn import geron_kmeans

    X = np.vstack([lcg_matrix(15, 2, seed=1) * 0.5, lcg_matrix(15, 2, seed=2) * 0.5 + 8])
    r = geron_kmeans(X, n_clusters=2, seed=0)
    hand = 0.0
    for j in range(2):
        pts = X[r["labels"] == j]
        hand += float(np.sum((pts - pts.mean(axis=0)) ** 2))
    assert r["inertia"] == pytest.approx(hand)
    # centres really are the cluster means
    for j in range(2):
        assert np.allclose(r["centers"][j], X[r["labels"] == j].mean(axis=0))


def test_kmeans_inertia_falls_monotonically_with_k():
    from morie.fn.hmkmn import geron_kmeans

    X = lcg_matrix(24, 2, seed=404) * 10
    inertias = [geron_kmeans(X, n_clusters=k, seed=0)["inertia"] for k in range(1, 6)]
    assert np.all(np.diff(inertias) <= 1e-9)
    assert inertias[0] == pytest.approx(float(np.sum((X - X.mean(axis=0)) ** 2)))


def test_kmeans_pp_seeds_are_distinct_points_of_the_data():
    from morie.fn.hmkmpp import geron_kmeans_plus_plus

    X = lcg_matrix(20, 2, seed=505)
    r = geron_kmeans_plus_plus(X, n_clusters=4, seed=3)
    assert len(set(int(i) for i in r["indices"])) == 4
    for c in r["centers"]:
        assert np.any(np.all(np.isclose(X, c), axis=1))


def test_minibatch_kmeans_inertia_is_not_better_than_exact():
    from morie.fn.hmkmn import geron_kmeans
    from morie.fn.hmmbkm import geron_minibatch_kmeans

    X = np.vstack([lcg_matrix(20, 1, seed=7), lcg_matrix(20, 1, seed=8) + 12])
    exact = geron_kmeans(X, n_clusters=2, seed=0)["inertia"]
    mb = geron_minibatch_kmeans(X, n_clusters=2, batch_size=5, seed=0, n_iter=400)["inertia"]
    assert mb >= exact - 1e-9


def test_image_segmentation_replaces_pixels_with_their_palette_entry():
    from morie.fn.hmiseg import geron_image_segmentation

    img = np.zeros((4, 4, 3))
    img[:2] = 1.0
    r = geron_image_segmentation(img, n_clusters=2, seed=0)
    assert r["inertia"] == pytest.approx(0.0)
    lab = r["labels"]
    for i in range(4):
        for j in range(4):
            assert np.allclose(r["segmented"][i, j], r["palette"][lab[i, j]])


def test_kmeans_limits_flags_anisotropy_and_size_imbalance():
    from morie.fn.hmkmlim import geron_kmeans_limits

    t = np.linspace(0, 1, 12)
    Y = np.vstack([np.c_[t * 12, np.zeros(12)], np.c_[t * 12, np.full(12, 9.0)]])
    r = geron_kmeans_limits(Y, n_clusters=2, seed=0)
    assert r["max_anisotropy"] > 10
    tight = np.vstack([lcg_matrix(10, 2, seed=1) * 0.1, lcg_matrix(10, 2, seed=2) * 0.1 + 20])
    s = geron_kmeans_limits(tight, n_clusters=2, seed=0)
    assert s["size_ratio"] == pytest.approx(1.0)
    assert s["reassigned_fraction"] == 0.0


def test_mean_shift_bandwidth_controls_the_cluster_count():
    from morie.fn.hmmnsh import geron_mean_shift

    X = np.array([[0.0], [0.2], [5.0], [5.2], [10.0], [10.2]])
    assert geron_mean_shift(X, bandwidth=0.5)["n_clusters"] == 3
    assert geron_mean_shift(X, bandwidth=1000.0)["n_clusters"] == 1
    wide = geron_mean_shift(X, bandwidth=1000.0)
    assert float(wide["modes"][0, 0]) == pytest.approx(float(X.mean()))


# ── dimensionality reduction: hmipca, hmkprbf/poly/sigmoid, hmmds, ------
# ── hmiso, hmlle -------------------------------------------------------


def test_incremental_pca_is_batch_size_invariant_and_matches_full_pca():
    from morie.fn.hmipca import geron_incremental_pca

    X = lcg_matrix(40, 4, seed=909) * np.array([5.0, 1.0, 0.2, 3.0])
    full = geron_incremental_pca(X, n_components=4)
    for bs in (1, 3, 7, 40):
        part = geron_incremental_pca(X, n_components=4, batch_size=bs)
        assert np.allclose(part["explained_variance"], full["explained_variance"])
        assert np.allclose(part["mean"], X.mean(axis=0))
    # independent route: eigenvalues of np.cov
    ev = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
    assert np.allclose(full["explained_variance"], ev)


def test_linear_kernel_pca_reproduces_ordinary_pca_variance():
    from morie.fn.hmkppl import geron_kernel_pca_poly

    X = lcg_matrix(15, 3, seed=1212) * 4
    r = geron_kernel_pca_poly(X, n_components=3, degree=1, gamma=1.0, coef0=0.0)
    # eigenvalues of the centred Gram matrix equal (m-1) * PCA eigenvalues
    pca_ev = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
    assert np.allclose(np.sort(r["eigenvalues"])[::-1] / (15 - 1), pca_ev, atol=1e-8)


def test_rbf_kernel_pca_gram_is_psd_with_unit_diagonal():
    from morie.fn.hmkprbf import geron_kernel_pca_rbf

    X = lcg_matrix(12, 2, seed=1313) * 3
    r = geron_kernel_pca_rbf(X, n_components=3, gamma=0.7)
    assert np.allclose(np.diag(r["K"]), 1.0)
    assert np.all(np.linalg.eigvalsh(r["K"]) > -1e-9)
    assert np.allclose(np.sum(r["X_projected"], axis=0), 0.0, atol=1e-9)
    # kernel entries checked by hand
    d2 = float(np.sum((X[0] - X[1]) ** 2))
    assert r["K"][0, 1] == pytest.approx(math.exp(-0.7 * d2))


def test_sigmoid_kernel_pca_reports_its_non_psd_spectrum():
    from morie.fn.hmkpsg import geron_kernel_pca_sigmoid

    X = lcg_matrix(10, 2, seed=1414) * 6 - 3
    r = geron_kernel_pca_sigmoid(X, n_components=2, gamma=1.0, coef0=0.0)
    assert np.all(np.abs(r["K"]) <= 1.0)
    assert r["K"][0, 1] == pytest.approx(math.tanh(float(X[0] @ X[1])))
    assert r["is_psd"] == (r["n_negative_eigenvalues"] == 0)


def test_mds_recovers_distances_and_flags_non_euclidean_input():
    from morie.fn.hmmds import geron_mds, pairwise_distances

    X = lcg_matrix(10, 3, seed=1515) * 5
    D = pairwise_distances(X)
    r = geron_mds(D, n_components=3, precomputed=True)
    assert np.allclose(pairwise_distances(r["embedding"]), D, atol=1e-8)
    assert r["stress"] == pytest.approx(0.0, abs=1e-16)
    bad = np.array([[0.0, 1.0, 9.0], [1.0, 0.0, 1.0], [9.0, 1.0, 0.0]])
    assert geron_mds(bad, n_components=1, precomputed=True)["n_negative_eigenvalues"] >= 1


def test_isomap_geodesic_exceeds_the_chord_on_a_curve():
    from morie.fn.hmiso import geron_isomap
    from morie.fn.hmmds import pairwise_distances

    theta = np.linspace(0, math.pi, 12)
    C = np.c_[np.cos(theta), np.sin(theta)]  # half circle, radius 1
    r = geron_isomap(C, n_components=2, n_neighbors=2)
    chord = float(pairwise_distances(C)[0, -1])
    geo = float(r["geodesic_distances"][0, -1])
    assert chord == pytest.approx(2.0)
    assert geo > chord
    assert geo == pytest.approx(math.pi, rel=0.02)  # arc length


def test_isomap_refuses_a_disconnected_graph():
    from morie.fn.hmiso import geron_isomap

    with pytest.raises(ValueError, match="disconnected"):
        geron_isomap([[0.0], [0.1], [50.0], [50.1]], n_components=1, n_neighbors=1)


def test_lle_weights_sum_to_one_and_reconstruct_interior_points():
    from morie.fn.hmlle import geron_locally_linear_embedding

    X = np.c_[np.linspace(0, 5, 10), np.zeros(10)]
    r = geron_locally_linear_embedding(X, n_components=1, n_neighbors=2)
    assert np.allclose(np.sum(r["weights"], axis=1), 1.0)
    assert np.max(r["reconstruction_error"][1:-1]) < 1e-12
    emb = r["embedding"][:, 0]
    order = np.argsort(emb)
    assert np.array_equal(order, np.arange(10)) or np.array_equal(order, np.arange(9, -1, -1))


# ── model selection: hmkfd, hmlcv, hmgrs, hmhpt, hmhplm, hmlrh ---------


def test_kfold_indices_form_a_partition():
    from morie.fn.hmkfd import geron_kfold

    X = lcg_matrix(11, 2, seed=1616)
    y = lcg(11, seed=1617)
    r = geron_kfold(X, y, k=4, seed=2)
    all_test = np.concatenate(r["test_indices"])
    assert sorted(all_test.tolist()) == list(range(11))
    for tr, te in zip(r["train_indices"], r["test_indices"]):
        assert set(tr.tolist()) & set(te.tolist()) == set()
        assert len(tr) + len(te) == 11
    assert max(r["fold_sizes"]) - min(r["fold_sizes"]) <= 1


def test_learning_curves_separate_underfit_from_good_fit():
    from morie.fn.hmlcv import geron_learning_curves

    x = np.linspace(0, 10, 40)
    X = np.c_[np.ones(40), x]
    y = 2.0 + 3.0 * x
    good = geron_learning_curves(X, y, n_splits=5, seed=0)
    assert good["rmse_val"][-1] < 1e-8
    bad = geron_learning_curves(np.ones((40, 1)), y, n_splits=5, seed=0)
    assert bad["verdict"] == "underfitting"
    assert bad["rmse_train"][-1] > 1.0


def test_grid_search_selects_by_cross_validated_score():
    from morie.fn.hmgrs import geron_grid_search

    X = np.c_[np.ones(12), np.linspace(0, 1, 12)]
    y = 1.0 + 2.0 * X[:, 1]
    r = geron_grid_search({"alpha": [0.0, 0.01, 10.0, 1000.0]}, X, y, K=3)
    assert r["best_params"] == {"alpha": 0.0}
    assert r["n_candidates"] == 4 and r["n_fits"] == 12
    scores = [c["cv_score"] for c in r["results"]]
    assert scores[0] == max(scores)


def test_random_search_costs_what_it_is_asked_to():
    from morie.fn.hmhpt import geron_hyperparameter_tuning

    X = np.c_[np.ones(12), np.linspace(0, 1, 12)]
    y = 1.0 + 2.0 * X[:, 1]
    grid = {"alpha": [0.0, 0.1, 1.0, 10.0, 100.0]}
    g = geron_hyperparameter_tuning(grid, X, y, K=2)
    s = geron_hyperparameter_tuning(grid, X, y, search="random", n_iter=3, K=2, seed=0)
    assert g["n_candidates"] == 5
    assert s["n_candidates"] == 3 and s["n_fits"] == 6


def test_depth_heuristic_returns_the_best_depth_not_the_last():
    from morie.fn.hmhplm import geron_hidden_layers_heuristic

    X = lcg_matrix(30, 2, seed=1818)
    y = lcg(30, seed=1819)
    vshape = lambda L, Xt, yt, Xv, yv: abs(L - 4) + 1.0
    r = geron_hidden_layers_heuristic(vshape, X, y, max_layers=12, patience=3)
    assert r["best_n_layers"] == 4
    assert r["stopped_early"] is True
    assert r["depths_tried"] < 12


def test_lr_finder_recommends_a_tenth_of_the_divergence_rate():
    from morie.fn.hmlrh import geron_learning_rate_heuristic

    lrs = [1e-5 * 10**i for i in range(7)]
    losses = [3.0, 2.0, 1.2, 0.6, 0.4, 8.0, 200.0]
    r = geron_learning_rate_heuristic(list(zip(lrs, losses)))
    assert r["lr_diverge"] == pytest.approx(lrs[5])
    assert r["lr"] == pytest.approx(lrs[5] / 10)
    assert r["lr_min_loss"] == pytest.approx(lrs[4])
    with pytest.raises(ValueError, match="increasing"):
        geron_learning_rate_heuristic([(1.0, 1.0), (0.1, 2.0)])


# ── attention / MLP / normalisation ------------------------------------


def test_multihead_attention_rows_are_distributions_and_scale_by_head_width():
    from morie.fn.hmmha import geron_multihead_attention

    Q = lcg_matrix(3, 4, seed=1919)
    K = lcg_matrix(5, 4, seed=1920)
    V = lcg_matrix(5, 4, seed=1921)
    r = geron_multihead_attention(Q, K, V, n_heads=2)
    assert r["d_head"] == 2
    for A in r["attention_weights"]:
        assert np.allclose(A.sum(axis=1), 1.0)
    # hand-recompute head 0 with the head width in the scale
    s = Q[:, :2] @ K[:, :2].T / math.sqrt(2)
    e = np.exp(s - s.max(axis=1, keepdims=True))
    A0 = e / e.sum(axis=1, keepdims=True)
    assert np.allclose(r["attention_weights"][0], A0)
    assert np.allclose(r["head_outputs"][0], A0 @ V[:, :2])


def test_multihead_attention_rejects_indivisible_head_count():
    from morie.fn.hmmha import geron_multihead_attention

    with pytest.raises(ValueError, match="does not divide"):
        geron_multihead_attention(lcg_matrix(2, 5, 1), lcg_matrix(2, 5, 2), lcg_matrix(2, 5, 3), n_heads=2)


def test_mlp_forward_matches_hand_matrix_products():
    from morie.fn.hmmlpf import geron_mlp

    W1 = np.array([[1.0, -2.0], [0.5, 0.25]])
    b1 = np.array([0.1, -0.1])
    W2 = np.array([[2.0], [-1.0]])
    b2 = np.array([0.5])
    X = np.array([[1.0, 2.0]])
    h = np.maximum(X @ W1 + b1, 0.0)
    out = h @ W2 + b2
    r = geron_mlp(X, [W1, W2], [b1, b2], ["relu", "identity"])
    assert np.allclose(r["output"], out)
    assert np.allclose(r["activations"][0], h)
    assert r["n_parameters"] == W1.size + b1.size + W2.size + b2.size


def test_layer_norm_gives_zero_mean_unit_variance_per_row():
    from morie.fn.hmlntr import geron_layer_normalization

    X = lcg_matrix(6, 5, seed=2020) * np.array([1.0, 10.0, 100.0, 0.1, 5.0])
    r = geron_layer_normalization(X, eps=0.0)
    assert np.allclose(r["x_hat"].mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(r["x_hat"].var(axis=1), 1.0)
    # gamma/beta applied per feature
    g = np.arange(1.0, 6.0)
    b = np.full(5, -2.0)
    a = geron_layer_normalization(X, gamma=g, beta=b, eps=0.0)
    assert np.allclose(a["y"], g * r["x_hat"] + b)


def test_layer_norm_rnn_normalises_each_time_step_separately():
    from morie.fn.hmlnr import geron_layer_norm_rnn

    x = np.array([[1.0, 3.0], [1000.0, 3000.0]])
    r = geron_layer_norm_rnn(x, eps=0.0, activation="none")
    assert np.allclose(r["normalized"][0], r["normalized"][1])
    assert np.allclose(r["mu"], [2.0, 2000.0])


# ── quantization / caches ----------------------------------------------


def test_int8_round_trip_error_never_exceeds_half_a_step():
    from morie.fn.hmint8 import geron_int8_quant

    x = lcg(200, seed=2121) * 6 - 3
    for bits in (4, 8, 12):
        r = geron_int8_quant(x, n_bits=bits)
        assert r["max_error"] <= r["scale"] / 2 + 1e-12
        assert np.all(np.abs(r["q"]) <= 2 ** (bits - 1) - 1)
    sym = geron_int8_quant([-2.0, 0.0, 2.0])
    assert float(sym["dequantized"][1]) == 0.0  # zero exactly representable


def test_int8_asymmetric_uses_the_whole_level_range():
    from morie.fn.hmint8 import geron_int8_quant

    x = lcg(50, seed=2222) * 5
    r = geron_int8_quant(x, n_bits=8, symmetric=False)
    assert int(r["q"].min()) == 0
    assert int(r["q"].max()) == 255
    assert r["scale"] == pytest.approx((x.max() - x.min()) / 255)


def test_kv_cache_per_head_scales_beat_a_global_scale():
    from morie.fn.hmkvc import geron_kv_cache_compress

    K = lcg_matrix(3 * 8 * 4, 1, seed=2323).reshape(3, 8, 4)
    K[0] *= 500.0
    per = geron_kv_cache_compress(K, K, n_bits=8, per_head=True)
    glob = geron_kv_cache_compress(K, K, n_bits=8, per_head=False)
    # the loud head sets the global scale, so the quiet heads lose almost
    # all their resolution under it and keep it under per-head scaling
    quiet_per = float(np.max(np.abs(per["K_dequantized"][1:] - K[1:])))
    quiet_glob = float(np.max(np.abs(glob["K_dequantized"][1:] - K[1:])))
    assert quiet_per < quiet_glob / 100
    # byte accounting reproduced by hand
    n_elem = 2 * K.size
    assert per["bytes_before"] == n_elem * 2
    assert per["bytes_after"] == n_elem + 6 * 4


# ── distillation / MLM / ICL -------------------------------------------


def test_distillation_loss_equals_hand_computed_terms():
    from morie.fn.hmkd import geron_knowledge_distillation

    tl = np.array([[2.0, 1.0, 0.0]])
    sl = np.array([[0.5, 0.5, 0.5]])
    T, alpha = 2.0, 0.3

    def soft(z, t):
        e = np.exp(np.asarray(z) / t)
        return e / e.sum()

    pt = soft(tl[0], T)
    ps = soft(sl[0], T)
    kl = float(np.sum(pt * (np.log(pt) - np.log(ps))))
    ph = soft(sl[0], 1.0)
    ce = -math.log(ph[0])
    hand = alpha * ce + (1 - alpha) * T * T * kl
    r = geron_knowledge_distillation(tl, sl, y=[0], T=T, alpha=alpha)
    assert r["kl_loss"] == pytest.approx(T * T * kl)
    assert r["ce_loss"] == pytest.approx(ce)
    assert r["loss"] == pytest.approx(hand)


def test_distillation_kl_is_zero_only_when_the_student_matches():
    from morie.fn.hmkd import geron_knowledge_distillation

    L = [[1.0, -1.0, 0.5]]
    assert geron_knowledge_distillation(L, L, T=3.0, alpha=0.0)["kl_loss"] == pytest.approx(0.0)
    off = geron_knowledge_distillation(L, [[1.0, -1.0, 0.6]], T=3.0, alpha=0.0)["kl_loss"]
    assert off > 0.0


def test_masked_lm_loss_scores_only_masked_positions():
    from morie.fn.hmmlm import geron_masked_lm

    X = [0, 1, 2, 3] * 5
    uniform = lambda mx, pos: np.full((len(pos), 4), 0.25)
    u = geron_masked_lm(X, mask_frac=0.25, seed=1, model=uniform, vocab_size=4)
    assert u["loss"] == pytest.approx(math.log(4))
    assert u["perplexity"] == pytest.approx(4.0)
    assert u["n_masked"] == 5
    # positions carry the true targets, and the masked copy hides them
    for p, t in zip(u["masked_positions"], u["targets"]):
        assert X[p] == t
        assert u["masked_input"].ravel()[p] == -1


def test_masked_lm_rejects_non_distribution_output():
    from morie.fn.hmmlm import geron_masked_lm

    with pytest.raises(ValueError, match="sum to 1"):
        geron_masked_lm([0, 1, 2, 3] * 5, model=lambda mx, pos: np.full((len(pos), 4), 0.4), vocab_size=4)


def test_in_context_learning_builds_the_prompt_and_normalises_scores():
    from morie.fn.hmicl import geron_in_context_learning

    ex = [("x1", "A"), ("x2", "B")]
    seen = {}

    def scorer(prompt, cand):
        seen[cand] = prompt
        return {"A": -1.0, "B": -3.0}[cand]

    r = geron_in_context_learning(scorer, ex, "q")
    assert r["prediction"] == "A"
    assert r["prompt"] == "x1 -> A\nx2 -> B\nq ->"
    assert set(seen) == {"A", "B"}
    # posterior is the softmax of the log-probs
    z = np.array([-1.0, -3.0])
    e = np.exp(z - z.max())
    assert np.allclose(r["posterior"], e / e.sum())


# ── pipelines / trainer / MCP ------------------------------------------


def test_pipeline_classification_softmax_and_contract():
    from morie.fn.hmhfpi import geron_hf_pipelines

    logits = [[1.0, 3.0], [4.0, 0.0]]
    r = geron_hf_pipelines("text-classification", ["a", "b"], lambda xs: logits, labels=["N", "P"])
    e0 = np.exp(np.array([1.0, 3.0]) - 3.0)
    assert r["predictions"][0]["score"] == pytest.approx(float((e0 / e0.sum())[1]))
    assert [p["label"] for p in r["predictions"]] == ["P", "N"]
    with pytest.raises(ValueError, match="rows for"):
        geron_hf_pipelines("text-classification", ["a", "b"], lambda xs: [[1.0, 0.0]])


def test_trainer_selects_the_best_eval_checkpoint():
    from morie.fn.hmhftn import geron_hf_trainer

    X = np.c_[np.ones(20), np.linspace(0, 1, 20)]
    y = 1.0 + 2.0 * X[:, 1]

    def lg(p, Xb, yb):
        r = Xb @ p - yb
        return float(np.mean(r**2)), (2.0 / len(yb)) * (Xb.T @ r)

    out = geron_hf_trainer({"params": np.zeros(2), "loss_and_grad": lg},
                           {"epochs": 400, "batch_size": 5, "learning_rate": 0.1, "seed": 0},
                           (X, y), (X, y))
    assert np.allclose(out["params"], [1.0, 2.0], atol=1e-2)
    best = min(h["eval_loss"] for h in out["history"])
    assert out["eval_loss"] == pytest.approx(best)
    assert len(out["history"]) == 400


def test_trainer_enforces_the_gradient_shape_contract():
    from morie.fn.hmhftn import geron_hf_trainer

    X = np.ones((4, 2))
    y = np.zeros(4)
    with pytest.raises(ValueError, match="gradient of shape"):
        geron_hf_trainer({"params": np.zeros(2), "loss_and_grad": lambda p, a, b: (1.0, np.zeros(5))},
                         {"epochs": 1, "batch_size": 4}, (X, y))


def test_mcp_requires_matching_ids_and_exactly_one_of_result_or_error():
    from morie.fn.hmmcp import geron_model_context_protocol

    ok = lambda req: {"jsonrpc": "2.0", "id": req["id"], "result": {"n": 1}}
    reqs = [{"jsonrpc": "2.0", "id": i, "method": "tools/list"} for i in (1, 2, 3)]
    r = geron_model_context_protocol(ok, reqs)
    assert r["n_ok"] == 3 and r["n_errors"] == 0
    with pytest.raises(ValueError, match="mismatched id"):
        geron_model_context_protocol(lambda req: {"jsonrpc": "2.0", "id": 42, "result": {}}, reqs[:1])
    with pytest.raises(ValueError, match="exactly one"):
        geron_model_context_protocol(lambda req: {"jsonrpc": "2.0", "id": req["id"]}, reqs[:1])
    with pytest.raises(ValueError, match="jsonrpc"):
        geron_model_context_protocol(ok, [{"jsonrpc": "1.0", "id": 1, "method": "x"}])


# ── uncertainty / anomalies / GAN diagnostics --------------------------


def test_mc_dropout_mean_is_unbiased_under_inverted_scaling():
    from morie.fn.hmmcd import geron_mc_dropout

    f = lambda z: np.asarray([float(np.sum(z))])
    total = 1.0 + 2.0 + 3.0 + 4.0
    r = geron_mc_dropout(f, [1.0, 2.0, 3.0, 4.0], K=20000, p=0.5, seed=0)
    assert abs(float(r["mean"][0]) - total) < 0.15
    assert float(r["var"][0]) > 0
    assert np.allclose(r["sem"], r["std"] / math.sqrt(20000))


def test_lof_scores_a_clear_outlier_above_the_cluster():
    from morie.fn.hmlof import geron_local_outlier_factor

    X = np.r_[lcg(20, seed=2424).reshape(-1, 1), [[25.0]]]
    r = geron_local_outlier_factor(X, n_neighbors=5)
    assert r["lof"][-1] == max(r["lof"])
    assert r["lof"][-1] > 2.0
    assert np.median(r["lof"][:-1]) < 1.5


def test_lof_is_relative_not_absolute_density():
    from morie.fn.hmlof import geron_local_outlier_factor

    dense = np.linspace(0, 1, 10).reshape(-1, 1)
    sparse = np.linspace(100, 200, 10).reshape(-1, 1)
    r = geron_local_outlier_factor(np.r_[dense, sparse], n_neighbors=3)
    # the sparse cluster is 100x less dense in absolute terms but is not flagged
    assert np.max(r["lof"][10:]) < 2.0


def test_mode_collapse_score_and_coverage_by_hand():
    from morie.fn.hmmdc import geron_mode_collapse

    same = geron_mode_collapse([[2.0]] * 5)
    assert same["n_modes"] == 1
    assert same["collapse_score"] == pytest.approx(1 - 1 / 5)
    distinct = geron_mode_collapse([[0.0], [10.0], [20.0], [30.0], [40.0]])
    assert distinct["n_modes"] == 5 and distinct["collapse_score"] == 0.0
    real = [[0.0], [10.0], [20.0], [30.0]]
    half = geron_mode_collapse([[0.05], [10.05], [0.1], [9.95]], reference=real)
    assert half["coverage"] == pytest.approx(0.5)


# ── MDP -----------------------------------------------------------------


def test_mdp_value_iteration_matches_the_geometric_series():
    from morie.fn.hmmdp import geron_mdp

    for g in (0.0, 0.5, 0.9, 0.99):
        r = geron_mdp(["s"], ["go"], [[[1.0]]], [[2.0]], gamma=g, max_iter=100000)
        assert float(r["V"][0]) == pytest.approx(2.0 / (1 - g), rel=1e-6)


def test_mdp_two_state_optimal_policy_by_hand():
    from morie.fn.hmmdp import geron_mdp

    # state 0: "stay" self-loops for 0, "move" goes to state 1 for 1
    # state 1: absorbing under both actions, reward 1 per step
    P = [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]]
    R = [[0.0, 1.0], [1.0, 1.0]]
    r = geron_mdp([0, 1], ["stay", "move"], P, R, gamma=0.9, max_iter=100000)
    assert int(r["policy"][0]) == 1  # "move" strictly dominates in state 0
    assert float(r["V"][1]) == pytest.approx(1 / 0.1, rel=1e-6)
    assert float(r["V"][0]) == pytest.approx(1 + 0.9 * 10, rel=1e-6)
    # both actions are identical in state 1, so its Q row is flat
    assert float(r["Q"][1, 0]) == pytest.approx(float(r["Q"][1, 1]))


def test_mdp_rejects_leaking_transitions_and_gamma_one():
    from morie.fn.hmmdp import geron_mdp

    with pytest.raises(ValueError, match="sums to"):
        geron_mdp(["s"], ["a"], [[[0.5]]], [[1.0]], gamma=0.9)
    with pytest.raises(ValueError, match="contraction"):
        geron_mdp(["s"], ["a"], [[[1.0]]], [[1.0]], gamma=1.0)


# ── multilabel / softmax regression / instance vs model based ----------


def test_multilabel_metrics_by_brute_force_counting():
    from morie.fn.hmmlb import geron_multilabel

    Y = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    P = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0]])
    r = geron_multilabel(np.arange(3).reshape(-1, 1).astype(float), Y, Y_pred=P)
    wrong = int(np.sum(Y != P))
    assert r["hamming_loss"] == pytest.approx(wrong / Y.size)
    assert r["subset_accuracy"] == pytest.approx(1 / 3)
    jac = []
    for i in range(3):
        inter = int(np.sum((Y[i] == 1) & (P[i] == 1)))
        union = int(np.sum((Y[i] == 1) | (P[i] == 1)))
        jac.append(inter / union)
    assert r["jaccard"] == pytest.approx(float(np.mean(jac)))


def test_softmax_regression_gradient_matches_finite_difference():
    from morie.fn.hmmnl import geron_multinomial_logistic

    X = lcg_matrix(12, 2, seed=2525) * 3
    y = np.array([0, 1, 2] * 4)

    def loss_at(Theta_flat):
        Theta = np.asarray(Theta_flat).reshape(3, 3)
        Xb = np.hstack([np.ones((12, 1)), X])
        Z = Xb @ Theta
        Z = Z - Z.max(axis=1, keepdims=True)
        P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
        return float(np.mean(-np.log(P[np.arange(12), y])))

    Theta = lcg_matrix(3, 3, seed=2526)
    Xb = np.hstack([np.ones((12, 1)), X])
    Z = Xb @ Theta
    Z = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
    Yh = np.eye(3)[y]
    analytic = (Xb.T @ (P - Yh)) / 12
    flat = Theta.ravel()
    for i in range(9):
        assert central_diff(loss_at, flat, i) == pytest.approx(analytic.ravel()[i], abs=1e-5)
    # and the fit itself separates separable classes
    Xs = np.array([[0.0], [0.2], [5.0], [5.2], [10.0], [10.2]])
    ys = [0, 0, 1, 1, 2, 2]
    r = geron_multinomial_logistic(Xs, ys, lr=0.5, n_iter=3000)
    assert r["accuracy"] == 1.0
    assert r["loss_history"][0] == pytest.approx(math.log(3))


def test_knn_prediction_matches_brute_force_neighbour_search():
    from morie.fn.hmins import geron_instance_based

    X = lcg_matrix(30, 2, seed=2727) * 10
    y = (X[:, 0] > 5).astype(int)
    q = np.array([[4.0, 4.0]])
    r = geron_instance_based(X, y, q, k=5)
    d = np.sqrt(np.sum((X - q) ** 2, axis=1))
    nearest = np.argsort(d)[:5]
    votes = np.bincount(y[nearest])
    assert int(r["prediction"][0]) == int(np.argmax(votes))
    assert sorted(r["neighbors"][0].tolist()) == sorted(nearest.tolist())


def test_model_based_fit_matches_the_normal_equations():
    from morie.fn.hmmod import geron_model_based

    X = lcg_matrix(25, 2, seed=2828) * 6 - 3
    y = 4.0 - 1.5 * X[:, 0] + 0.75 * X[:, 1]
    r = geron_model_based(X, y, n_iter=20000)
    Xb = np.hstack([np.ones((25, 1)), X])
    cf = np.linalg.lstsq(Xb, y, rcond=None)[0]
    assert np.allclose(r["theta"], cf)
    assert np.allclose(r["theta"], [4.0, -1.5, 0.75], atol=1e-8)
    assert r["gap"] < 1e-6
    assert r["r2"] == pytest.approx(1.0)


# ── architecture entries: hmlnet, hmkrn --------------------------------


def test_lenet5_layer_shapes_and_parameter_counts_by_hand():
    from morie.fn.hmlnet import geron_lenet5

    r = geron_lenet5(n_classes=10)
    counts = {L["name"]: L["parameters"] for L in r["layers"]}
    assert counts["C1"] == 5 * 5 * 1 * 6 + 6
    assert counts["C3"] == 5 * 5 * 6 * 16 + 16
    assert counts["C5"] == 5 * 5 * 16 * 120 + 120
    assert counts["F6"] == 120 * 84 + 84
    assert counts["output"] == 84 * 10 + 10
    assert r["total_parameters"] == sum(counts.values())
    shapes = [L["output_shape"] for L in r["layers"]]
    assert [s[0] for s in shapes[:5]] == [28, 14, 10, 5, 1]


def test_lenet5_only_the_head_depends_on_the_class_count():
    from morie.fn.hmlnet import geron_lenet5

    a = geron_lenet5(n_classes=10)["total_parameters"]
    b = geron_lenet5(n_classes=100)["total_parameters"]
    assert b - a == 90 * 84 + 90


def test_conv_filter_parameters_are_independent_of_image_size():
    from morie.fn.hmkrn import geron_filter_kernel

    r = geron_filter_kernel(3, 3, 16, 32, seed=1)
    assert r["shape"] == (3, 3, 16, 32)
    assert r["n_parameters"] == 3 * 3 * 16 * 32 + 32
    assert r["fan_in"] == 3 * 3 * 16
    assert r["std"] == pytest.approx(math.sqrt(2 / (3 * 3 * 16)))
    # empirical spread agrees with the He target
    assert abs(float(np.std(r["kernel"])) - r["std"]) < 0.1 * r["std"]


# ── mean-of-inputs stubs must fail -------------------------------------


@pytest.mark.parametrize(
    "modname, fname",
    [
        ("hmmae", "geron_mae"),
        ("hmmms", "geron_min_max_scaling"),
        ("hmmom", "geron_momentum"),
        ("hmlogp", "geron_logistic_probability"),
        ("hmigr", "geron_information_gain"),
        ("hmint8", "geron_int8_quant"),
        ("hmjl", "geron_johnson_lindenstrauss"),
        ("hmhev", "geron_heaviside"),
    ],
)
def test_estimate_is_not_the_mean_of_the_first_argument(modname, fname):
    """A stub whose body is ``mean(first_arg)`` would pass a smoke test.

    Each call below is chosen so the correct ``estimate`` differs from the
    mean of the first argument, which is what the placeholder bodies
    returned.
    """
    import importlib

    mod = importlib.import_module("morie.fn." + modname)
    fn = getattr(mod, fname)
    calls = {
        "hmmae": (([10.0, 20.0, 30.0], [10.0, 20.0, 30.0]), {}),
        "hmmms": (([1.0, 2.0, 9.0],), {}),
        "hmmom": (([4.0, 4.0],), {"beta": 0.0, "eta": 1.0}),
        "hmlogp": (([[0.0], [0.0]], [7.0]), {}),
        "hmigr": (([0, 0, 1, 1], [0, 0, 1, 1]), {}),
        "hmint8": (([1.0, 2.0, 3.0],), {}),
        "hmjl": ((100, 0.3), {}),
        "hmhev": (([-5.0, -5.0, -5.0],), {}),
    }
    args, kwargs = calls[modname]
    r = fn(*args, **kwargs)
    first = np.atleast_1d(np.asarray(args[0], dtype=float))
    assert float(r["estimate"]) != pytest.approx(float(np.mean(first)))
    assert "method" in r and "n" in r and "estimate" in r


def test_every_module_returns_the_required_payload_keys():
    """estimate / n / method are part of the house contract -- checked
    by importing each module from the installed package, not by
    globbing a sandbox directory layout."""
    import importlib
    names = [l.split(" | ")[0] for l in open(
        "/Users/socialscientistlawyer/.claude/jobs/804e7f0b/tmp/w4b.txt")]
    assert len(names) >= 70
    for n in names:
        m = importlib.import_module(f"morie.fn.{n}")
        assert hasattr(m, "cheatsheet")

