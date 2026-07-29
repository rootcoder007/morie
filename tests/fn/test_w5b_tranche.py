"""Independent-route checks for the w5b tranche (Geron training mechanics / DL).

Every expected value here is derived from the formula by hand or from a
second, structurally different computation -- never by calling the module
and pasting what it printed. Deterministic data only: explicit lists or
the house LCG.
"""

import math

import numpy as np
import pytest

from morie.fn.grn016 import geron_ch4_logistic_regression_prediction
from morie.fn.grn021 import geron_ch4_softmax_function
from morie.fn.grn024 import geron_ch4_cross_entropy_gradient_vector
from morie.fn.grnag import geron_nesterov_accelerated_gradient
from morie.fn.grnmfo import geron_nmf_objective
from morie.fn.grnorm import geron_normal_equation
from morie.fn.grnsp import geron_bert_nsp_loss
from morie.fn.grnud import geron_numerical_differentiation
from morie.fn.groft import geron_overfitting_gap
from morie.fn.grohe import geron_one_hot_encoding
from morie.fn.grord import geron_ordinal_encoding
from morie.fn.grovo import geron_one_vs_one
from morie.fn.grovr import geron_one_vs_rest
from morie.fn.grpcap import geron_pca_projection
from morie.fn.grpe import geron_sinusoidal_positional_encoding
from morie.fn.grpels import geron_peephole_lstm_cell
from morie.fn.grpex import geron_prioritized_experience_weight
from morie.fn.grpio import geron_perceiver_io
from morie.fn.grpoly import geron_polynomial_features
from morie.fn.grppo import geron_ppo_clipped_objective
from morie.fn.grprc import geron_precision_recall_curve
from morie.fn.grpre import geron_precision
from morie.fn.grprn import geron_weight_pruning
from morie.fn.grptq import geron_static_ptq
from morie.fn.grpvt import geron_pyramid_vit_stage
from morie.fn.grq8 import geron_int8_quantization
from morie.fn.grqat import geron_quantization_aware_training
from morie.fn.grql import geron_q_learning_update
from morie.fn.grqpi import geron_action_value_function
from morie.fn.grrad import geron_reverse_mode_autodiff
from morie.fn.grrec import geron_recall
from morie.fn.grrein import geron_reinforce_policy_gradient
from morie.fn.grrep import geron_reparameterization_trick
from morie.fn.grret import geron_discounted_return
from morie.fn.grridg import geron_ridge_cost
from morie.fn.grridn import geron_ridge_normal_equation
from morie.fn.grrlhf import geron_rlhf_reward_kl_objective
from morie.fn.grrmse import geron_rmse
from morie.fn.grrnd import geron_randomized_search_cv
from morie.fn.grrnnc import geron_simple_rnn_cell
from morie.fn.grroc import geron_roc_curve
from morie.fn.grrsk import geron_resnet_skip
from morie.fn.grsa import geron_self_attention
from morie.fn.grsae import geron_sparse_autoencoder
from morie.fn.grscm import geron_score_matching_loss
from morie.fn.grsdpa import geron_scaled_dot_product_attention
from morie.fn.grsen import geron_senet_squeeze_excite
from morie.fn.grsft import geron_sft_objective
from morie.fn.grsgd import geron_stochastic_gradient_descent
from morie.fn.grsig import geron_sigmoid
from morie.fn.grsil import geron_silhouette_score
from morie.fn.grsmd import geron_symbolic_differentiation
from morie.fn.grsmxp import geron_softmax_probability
from morie.fn.grsmxs import geron_softmax_score
from morie.fn.grsnt import geron_sentiment_binary
from morie.fn.grstae import geron_stacked_autoencoder
from morie.fn.grstd import geron_standardization
from morie.fn.grstk import geron_stacking_predictor
from morie.fn.grswin import geron_swin_window_attention
from morie.fn.grtd0 import geron_td_zero_update
from morie.fn.grtdb import geron_transformer_decoder_block
from morie.fn.grteb import geron_transformer_encoder_block
from morie.fn.grtlu import geron_threshold_logic_unit
from morie.fn.grtmp import geron_temperature_sampling
from morie.fn.grtnh import geron_tanh_activation
from morie.fn.grtop import geron_topk_sampling
from morie.fn.grtrc import geron_tree_classification_leaf
from morie.fn.grtrv import geron_tree_regression_leaf
from morie.fn.grvae import geron_vae_elbo
from morie.fn.grvit import geron_vit_patch_embedding
from morie.fn.grvoth import geron_hard_voting
from morie.fn.grvots import geron_soft_voting
from morie.fn.grvpi import geron_state_value_function
from morie.fn.grwdc import geron_adamw_decoupled_weight_decay
from morie.fn.grwpc import geron_wordpiece_tokenizer_score
from morie.fn.grxeng import geron_softmax_cost_gradient
from morie.fn.grxent import geron_softmax_cross_entropy_cost
from morie.fn.grxgbg import geron_xgboost_gain
from morie.fn.grxvi import geron_glorot_xavier_init
from morie.fn.gryol import geron_yolo_grid_loss

TOL = 1e-9


def lcg(n, seed=2024):
    """House LCG: s = (1664525 s + 1013904223) mod 2^32, u = (s + 0.5)/2^32."""
    s = seed % 2**32
    out = []
    for _ in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out.append((s + 0.5) / 2**32)
    return out


# ---------------------------------------------------------------- activations


def test_grsig_matches_exponential_definition():
    t = [-3.0, -0.5, 0.0, 0.5, 3.0]
    want = [1.0 / (1.0 + math.exp(-v)) for v in t]
    got = geron_sigmoid(t)["sigma"]
    assert np.allclose(got, want, atol=TOL)
    # A mean-of-inputs stub would return mean(t) = 0.0, not this vector.
    assert not np.isclose(got[0], float(np.mean(t)))


def test_grsig_saturates_without_overflow():
    r = geron_sigmoid([-1000.0, 1000.0])
    assert r["sigma"] == [0.0, 1.0]
    assert r["derivative"] == [0.0, 0.0]


def test_grtnh_matches_ratio_of_exponentials():
    z = [-2.0, -0.25, 0.0, 0.75, 2.0]
    want = [(math.exp(v) - math.exp(-v)) / (math.exp(v) + math.exp(-v)) for v in z]
    r = geron_tanh_activation(z)
    assert np.allclose(r["activation"], want, atol=TOL)
    assert np.allclose(r["derivative"], [1 - w * w for w in want], atol=TOL)


def test_grtlu_is_a_halfspace_indicator():
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    w, b = [1.0, 1.0], -1.5
    want = [1 if (x[0] * w[0] + x[1] * w[1] + b) >= 0 else 0 for x in X]
    assert geron_threshold_logic_unit(X, w, b)["output"] == want == [0, 0, 0, 1]


# ------------------------------------------------------------ softmax family


def test_grn021_softmax_against_manual_exponentials():
    s = [0.0, 1.0, 2.0]
    denom = sum(math.exp(v) for v in s)
    for k in range(3):
        r = geron_ch4_softmax_function(s, k=k, K=3)
        assert abs(r["probability"] - math.exp(s[k]) / denom) < TOL


def test_grn021_is_shift_invariant_and_rejects_bad_k():
    a = geron_ch4_softmax_function([1.0, 2.0], 0)["probability"]
    b = geron_ch4_softmax_function([101.0, 102.0], 0)["probability"]
    assert abs(a - b) < 1e-12
    with pytest.raises(ValueError):
        geron_ch4_softmax_function([1.0, 2.0], 5)


def test_grsmxs_scores_are_the_matrix_product():
    X = [[1.0, 2.0], [3.0, 4.0]]
    T = [[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]]
    want = [[sum(X[i][f] * T[f][k] for f in range(2)) for k in range(3)] for i in range(2)]
    assert geron_softmax_score(X, T)["scores"] == want


def test_grsmxp_rows_are_probability_distributions():
    X = [[1.0, 2.0], [3.0, 4.0]]
    T = [[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]]
    P = geron_softmax_probability(X, T)["probabilities"]
    for row in P:
        assert abs(sum(row) - 1.0) < 1e-12
        assert all(p > 0 for p in row)
    # row 0 scores are [1, 2, 1]
    d = math.exp(1) + math.exp(2) + math.exp(1)
    assert abs(P[0][1] - math.exp(2) / d) < TOL


def test_grxent_equals_hand_log_sum():
    X = [[1.0], [1.0]]
    T = [[0.0, 1.0, 2.0]]
    d = math.exp(0) + math.exp(1) + math.exp(2)
    want = -(math.log(math.exp(0) / d) + math.log(math.exp(2) / d)) / 2
    got = geron_softmax_cross_entropy_cost(X, [0, 2], T)["cost"]
    assert abs(got - want) < TOL
    # mean-of-inputs stub would give mean(X) = 1.0
    assert abs(got - 1.0) > 0.1


def test_grxeng_gradient_matches_finite_differences():
    X = [[1.0, 2.0], [0.5, -1.0]]
    T = np.array([[0.3, -0.2, 0.1], [0.0, 0.4, -0.5]])
    Y = [0, 2]
    G = np.array(geron_softmax_cost_gradient(X, Y, T)["gradient"])
    h = 1e-6
    num = np.zeros_like(G)
    for i in range(T.shape[0]):
        for k in range(T.shape[1]):
            up, dn = T.copy(), T.copy()
            up[i, k] += h
            dn[i, k] -= h
            num[i, k] = (
                geron_softmax_cross_entropy_cost(X, Y, up)["cost"]
                - geron_softmax_cross_entropy_cost(X, Y, dn)["cost"]
            ) / (2 * h)
    assert np.allclose(G, num, atol=1e-6)


def test_grn024_selects_the_right_gradient_column():
    X = [[1.0, 2.0], [0.5, -1.0]]
    T = [[0.3, -0.2, 0.1], [0.0, 0.4, -0.5]]
    full = geron_softmax_cost_gradient(X, [0, 2], T)["gradient"]
    for k in range(3):
        col = geron_ch4_cross_entropy_gradient_vector(X, [0, 2], T, k)["gradient"]
        assert np.allclose(col, [row[k] for row in full], atol=TOL)


def test_grn016_threshold_boundary_and_validation():
    r = geron_ch4_logistic_regression_prediction([0.0, 0.4999, 0.5, 1.0])
    assert r["y_hat"] == [0, 0, 1, 1]
    with pytest.raises(ValueError):
        geron_ch4_logistic_regression_prediction([1.5])


# ------------------------------------------------------------- linear models


def test_grnorm_recovers_known_coefficients():
    x = [0.0, 1.0, 2.0, 3.0]
    X = [[1.0, v] for v in x]
    y = [4.0 + 3.0 * v for v in x]
    theta = geron_normal_equation(X, y)["theta"]
    assert np.allclose(theta, [4.0, 3.0], atol=1e-10)


def test_grnorm_raises_on_collinear_design():
    with pytest.raises(ValueError):
        geron_normal_equation([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], [1.0, 2.0, 3.0])


def test_grridn_solves_the_augmented_system_independently():
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([4.0, 7.0, 10.0])
    alpha = 3.0
    A = np.diag([0.0, 1.0])
    want = np.linalg.inv(X.T @ X + alpha * A) @ X.T @ y
    got = geron_ridge_normal_equation(X, y, alpha)["theta"]
    assert np.allclose(got, want, atol=1e-12)


def test_grridn_shrinks_slope_but_not_intercept_penalty():
    X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    y = [4.0, 7.0, 10.0]
    slopes = [abs(geron_ridge_normal_equation(X, y, a)["theta"][1]) for a in (0.0, 1.0, 100.0)]
    assert slopes[0] > slopes[1] > slopes[2]


def test_grridg_cost_and_gradient_agree_numerically():
    X = np.array([[1.0, 0.5], [1.0, -2.0], [1.0, 3.0]])
    y = np.array([1.0, 0.0, 2.0])
    th = np.array([0.2, -0.3])
    alpha = 0.7
    r = geron_ridge_cost(X, y, th, alpha)
    want = float(np.mean((X @ th - y) ** 2) + 0.5 * alpha * th[1] ** 2)
    assert abs(r["cost"] - want) < TOL
    h = 1e-7
    num = [
        (
            geron_ridge_cost(X, y, th + h * np.eye(2)[j], alpha)["cost"]
            - geron_ridge_cost(X, y, th - h * np.eye(2)[j], alpha)["cost"]
        )
        / (2 * h)
        for j in range(2)
    ]
    assert np.allclose(r["gradient"], num, atol=1e-6)


def test_grsgd_first_step_matches_hand_derivation():
    X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    y = [4.0, 7.0, 10.0]
    r = geron_stochastic_gradient_descent(X, y, [0.0, 0.0], eta=0.1, n_iter=1, seed=1)
    i = r["sample_order"][0]
    xi = np.array(X[i])
    grad = 2.0 * xi * (xi @ np.zeros(2) - y[i])
    assert np.allclose(r["theta"], -0.1 * grad, atol=1e-12)


def test_grsgd_is_seed_reproducible_and_descends():
    X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    y = [4.0, 7.0, 10.0]
    a = geron_stochastic_gradient_descent(X, y, [0.0, 0.0], 0.05, 150, seed=11)
    b = geron_stochastic_gradient_descent(X, y, [0.0, 0.0], 0.05, 150, seed=11)
    assert a["theta"] == b["theta"]
    assert a["cost_path"][-1] < a["cost_path"][0]


# --------------------------------------------------------------- metrics


def test_grrmse_against_manual_square_root():
    yt, yp = [1.0, 2.0, 3.0], [2.0, 1.0, 5.0]
    want = math.sqrt(sum((p - t) ** 2 for p, t in zip(yp, yt)) / 3)
    r = geron_rmse(yt, yp)
    assert abs(r["rmse"] - want) < TOL
    assert r["rmse"] >= r["mae"]
    assert abs(r["rmse"] - float(np.mean(yt))) > 0.1        # not a mean-of-inputs stub


def test_grpre_and_grrec_from_hand_counts():
    yt = [1, 1, 0, 0, 1]
    yp = [1, 0, 1, 0, 1]
    tp = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 0)
    assert abs(geron_precision(yt, yp)["precision"] - tp / (tp + fp)) < TOL
    assert abs(geron_recall(yt, yp)["recall"] - tp / (tp + fn)) < TOL


def test_grpre_raises_when_class_never_predicted():
    with pytest.raises(ValueError):
        geron_precision([1, 1, 0], [0, 0, 0])


def test_grroc_auc_equals_the_rank_statistic():
    y = [0, 1, 0, 1, 1]
    s = [0.1, 0.2, 0.3, 0.4, 0.35]
    pos = [s[i] for i in range(5) if y[i] == 1]
    neg = [s[i] for i in range(5) if y[i] == 0]
    want = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    assert abs(geron_roc_curve(y, s)["auc"] - want) < TOL


def test_grroc_monotone_curve_and_endpoints():
    r = geron_roc_curve([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert r["fpr"][0] == r["tpr"][0] == 0.0
    assert r["fpr"][-1] == r["tpr"][-1] == 1.0
    assert all(b >= a for a, b in zip(r["tpr"], r["tpr"][1:]))
    assert all(b >= a for a, b in zip(r["fpr"], r["fpr"][1:]))


def test_grprc_average_precision_by_hand():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    r = geron_precision_recall_curve(y, s)
    # thresholds 0.9, 0.8, 0.2, 0.1 -> precision 1, 1, 2/3, 1/2; recall .5, 1, 1, 1
    assert np.allclose(r["precision"], [1.0, 1.0, 2 / 3, 0.5], atol=TOL)
    assert np.allclose(r["recall"], [0.5, 1.0, 1.0, 1.0], atol=TOL)
    assert abs(r["average_precision"] - (0.5 * 1.0 + 0.5 * 1.0)) < TOL


def test_groft_gap_is_elementwise_difference():
    tr, va = [0.80, 0.90, 0.97], [0.78, 0.86, 0.80]
    r = geron_overfitting_gap(tr, va)
    assert np.allclose(r["gap"], [a - b for a, b in zip(tr, va)], atol=TOL)
    assert r["best_val_epoch"] == 1 and r["max_gap_epoch"] == 2


# ------------------------------------------------------- preprocessing


def test_grstd_output_has_zero_mean_unit_variance():
    u = lcg(20, seed=5)
    X = [[v, 3 * v + 1] for v in u]
    Z = np.array(geron_standardization(X)["scaled"])
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(Z.var(axis=0), 1.0, atol=1e-12)


def test_grstd_rejects_constant_column():
    with pytest.raises(ValueError):
        geron_standardization([[1.0, 5.0], [2.0, 5.0]])


def test_grohe_rows_are_indicators():
    r = geron_one_hot_encoding(["b", "a", "b", "c"])
    assert r["levels"] == ["a", "b", "c"]
    for row, cat in zip(r["encoded"], ["b", "a", "b", "c"]):
        assert sum(row) == 1.0
        assert row[r["levels"].index(cat)] == 1.0


def test_grord_respects_declared_order():
    r = geron_ordinal_encoding(["good", "bad", "average"], levels=["bad", "average", "good"])
    assert r["encoded"] == [2, 0, 1]
    with pytest.raises(ValueError):
        geron_ordinal_encoding(["x"], levels=["a", "b"])


def test_grpoly_columns_are_exact_powers():
    r = geron_polynomial_features([2.0, 3.0], degree=3)
    assert r["features"] == [[1.0, 2.0, 4.0, 8.0], [1.0, 3.0, 9.0, 27.0]]
    assert r["n_features"] == 4


# --------------------------------------------------------------- attention


def test_grsdpa_weights_match_manual_softmax():
    Q = [[1.0, 0.0]]
    K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[1.0, 0.0], [0.0, 1.0]]
    d = math.sqrt(2)
    e0, e1 = math.exp(1 / d), math.exp(0.0)
    want = [e0 / (e0 + e1), e1 / (e0 + e1)]
    r = geron_scaled_dot_product_attention(Q, K, V)
    assert np.allclose(r["weights"][0], want, atol=TOL)
    assert np.allclose(r["output"][0], want, atol=TOL)      # V is the identity


def test_grsdpa_mask_zeroes_blocked_keys():
    r = geron_scaled_dot_product_attention(
        [[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
        mask=[[True, False]],
    )
    assert r["weights"][0] == [1.0, 0.0]
    with pytest.raises(ValueError):
        geron_scaled_dot_product_attention(
            [[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
            mask=[[False, False]],
        )


def test_grsa_identity_projections_reduce_to_plain_attention():
    I = [[1.0, 0.0], [0.0, 1.0]]
    a = geron_self_attention(I, I, I, I)["weights"]
    b = geron_scaled_dot_product_attention(I, I, I)["weights"]
    assert np.allclose(a, b, atol=TOL)


def test_grsa_causal_mask_is_lower_triangular_in_effect():
    I = [[1.0, 0.0], [0.0, 1.0]]
    W = geron_self_attention(I, I, I, I, mask=[[True, False], [True, True]])["weights"]
    assert W[0][1] == 0.0
    assert abs(sum(W[1]) - 1.0) < 1e-12


def test_grteb_output_rows_are_layernormed():
    I = [[1.0, 0.0], [0.0, 1.0]]
    mha = {"WQ": [I], "WK": [I], "WV": [I], "WO": I}
    ffn = {"W1": [[0.0], [0.0]], "W2": [[0.0, 0.0]]}
    Y = np.array(geron_transformer_encoder_block(I, mha, ffn)["output"])
    assert np.allclose(Y.mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(Y.var(axis=1), 1.0, atol=1e-4)


def test_grtdb_builds_its_own_causal_mask():
    I = [[1.0, 0.0], [0.0, 1.0]]
    att = {"WQ": [I], "WK": [I], "WV": [I], "WO": I}
    W = {"self": att, "cross": att, "ffn": {"W1": [[0.0], [0.0]], "W2": [[0.0, 0.0]]}}
    r = geron_transformer_decoder_block(I, I, W)
    assert r["causal_mask"] == [[True, False], [True, True]]
    assert r["self_attention_weights"][0][0] == [1.0, 0.0]
    # cross-attention is NOT masked: token 0 sees both source positions
    assert all(w > 0 for w in r["cross_attention_weights"][0][0])


def test_grswin_window_size_one_is_pure_value_projection():
    X = [[[1.0], [2.0]], [[3.0], [4.0]]]
    I = [[1.0]]
    r = geron_swin_window_attention(X, 1, I, I, I)
    assert r["output"] == X
    assert r["n_windows"] == 4


def test_grswin_single_window_equals_global_attention():
    X = [[[1.0], [2.0]], [[3.0], [4.0]]]
    I = [[1.0]]
    flat = [[1.0], [2.0], [3.0], [4.0]]
    want = geron_scaled_dot_product_attention(flat, flat, flat)["output"]
    got = geron_swin_window_attention(X, 2, I, I, I)["output"]
    assert np.allclose(np.array(got).reshape(4, 1), want, atol=TOL)


def test_grpvt_full_reduction_gives_the_spatial_mean():
    X = [[[1.0], [2.0]], [[3.0], [4.0]]]
    I = [[1.0]]
    r = geron_pyramid_vit_stage(X, I, I, I, reduction_ratio=2)
    assert r["reduced_tokens"] == 1
    assert np.allclose(r["output"], [[2.5]] * 4, atol=TOL)


def test_grpvt_identity_reduction_matches_self_attention():
    X = [[[1.0], [2.0]], [[3.0], [4.0]]]
    I = [[1.0]]
    flat = [[1.0], [2.0], [3.0], [4.0]]
    want = geron_scaled_dot_product_attention(flat, flat, flat)["output"]
    got = geron_pyramid_vit_stage(X, I, I, I, reduction_ratio=1)["output"]
    assert np.allclose(got, want, atol=TOL)


def test_grpio_latent_bottleneck_and_shapes():
    X = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
    r = geron_perceiver_io(X, [[0.0, 0.0]], [[0.0, 0.0]])
    assert np.allclose(r["latent"][0], [2.5, 0.0], atol=TOL)   # uniform attention
    assert r["complexity_ratio"] == 4.0
    with pytest.raises(ValueError):
        geron_perceiver_io([[1.0]], [[0.0], [0.0]], [[0.0]])   # latent not smaller


def test_grpe_positional_encoding_matches_the_closed_form():
    r = geron_sinusoidal_positional_encoding(4, 4)
    PE = r["encoding"]
    for pos in range(4):
        for i in range(2):
            ang = pos / (10000 ** (2 * i / 4))
            assert abs(PE[pos][2 * i] - math.sin(ang)) < TOL
            assert abs(PE[pos][2 * i + 1] - math.cos(ang)) < TOL


def test_grpe_row_norms_are_constant():
    PE = np.array(geron_sinusoidal_positional_encoding(6, 8)["encoding"])
    norms = np.linalg.norm(PE, axis=1)
    assert np.allclose(norms, math.sqrt(4.0), atol=1e-12)
    with pytest.raises(ValueError):
        geron_sinusoidal_positional_encoding(3, 5)             # odd d_model


def test_grvit_patch_order_is_row_major():
    img = [[1.0, 2.0], [3.0, 4.0]]
    r = geron_vit_patch_embedding(img, 1, [[1.0]])
    assert r["embeddings"] == [[0.0], [1.0], [2.0], [3.0], [4.0]]
    with pytest.raises(ValueError):
        geron_vit_patch_embedding([[1.0, 2.0, 3.0]], 2, [[1.0]] * 4)


def test_grsen_gates_are_sigmoids_of_the_bottleneck():
    X = [[[1.0, 3.0]]]
    I = [[1.0, 0.0], [0.0, 1.0]]
    r = geron_senet_squeeze_excite(X, I, I)
    want = [1 / (1 + math.exp(-1.0)), 1 / (1 + math.exp(-3.0))]
    assert np.allclose(r["scale"], want, atol=TOL)
    assert np.allclose(r["output"][0][0], [1.0 * want[0], 3.0 * want[1]], atol=TOL)


# --------------------------------------------------------------- recurrent


def test_grrnnc_matches_manual_tanh_step():
    I = [[1.0, 0.0], [0.0, 1.0]]
    Wxh = [[2.0], [0.0]]
    r = geron_simple_rnn_cell([1.0], [0.5, -0.5], I, Wxh, [0.1, -0.1])
    want = [math.tanh(0.5 + 2.0 + 0.1), math.tanh(-0.5 + 0.0 - 0.1)]
    assert np.allclose(r["h"], want, atol=TOL)
    assert abs(r["spectral_norm_Whh"] - 1.0) < 1e-12


def test_grpels_gates_use_the_right_cell_state():
    W = [[0.0, 0.0]]
    r = geron_peephole_lstm_cell(
        [0.0], [0.0], [1.0], W, W, W, W, [2.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
    )
    f_want = 1 / (1 + math.exp(-2.0))          # peephole sees c_prev = 1
    i_want, g_want = 0.5, 0.0
    assert abs(r["f"][0] - f_want) < TOL
    assert abs(r["c"][0] - (f_want * 1.0 + i_want * g_want)) < TOL


def test_grpels_output_gate_peeps_at_the_new_state():
    W = [[0.0, 0.0]]
    r = geron_peephole_lstm_cell(
        [0.0], [0.0], [1.0], W, W, W, W, [0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0]
    )
    assert abs(r["c"][0] - 0.5) < TOL                    # f=i=0.5, g=0 -> c = 0.5
    assert abs(r["o"][0] - 1 / (1 + math.exp(-0.5))) < TOL


# ------------------------------------------------------ optimizers, init


def test_grnag_first_step_by_hand():
    r = geron_nesterov_accelerated_gradient([1.0], lambda t: 2 * t, [0.0], eta=0.1)
    # lookahead = 1 - 0.1*0.9*0 = 1; g = 2; v = 2; theta = 1 - 0.2
    assert r["lookahead"] == [1.0] and r["gradient"] == [2.0]
    assert abs(r["theta_new"][0] - 0.8) < TOL


def test_grnag_lookahead_differs_from_plain_momentum():
    r = geron_nesterov_accelerated_gradient([0.8], lambda t: 2 * t, [2.0], eta=0.1, beta=0.9)
    ahead = 0.8 - 0.1 * 0.9 * 2.0
    v = 0.9 * 2.0 + 2 * ahead
    assert abs(r["theta_new"][0] - (0.8 - 0.1 * v)) < TOL
    assert abs(r["lookahead"][0] - ahead) < TOL


def test_grnag_enforces_grad_fn_contract():
    with pytest.raises(ValueError):
        geron_nesterov_accelerated_gradient([1.0, 2.0], lambda t: [1.0], [0.0, 0.0], 0.1)


def test_grwdc_first_step_equals_adam_plus_decay():
    eta, lam = 0.001, 0.1
    r = geron_adamw_decoupled_weight_decay([1.0], [0.1], [0.0], [0.0], 1, eta, lam=lam)
    # bias-corrected first step of Adam has magnitude eta (g/|g| direction)
    assert abs(r["adam_step"][0] - eta * 0.1 / (math.sqrt(0.01) + 1e-8)) < 1e-9
    assert abs(r["decay_step"][0] - eta * lam * 1.0) < TOL
    assert abs(r["theta_new"][0] - (1.0 - r["adam_step"][0] - r["decay_step"][0])) < TOL


def test_grwdc_decay_is_not_routed_through_the_second_moment():
    # Two parameters, identical gradients, different magnitudes: the decay
    # must scale with theta, not with the normalised Adam step.
    r = geron_adamw_decoupled_weight_decay(
        [1.0, 10.0], [0.1, 0.1], [0.0, 0.0], [0.0, 0.0], 1, 0.01, lam=0.5
    )
    assert abs(r["adam_step"][0] - r["adam_step"][1]) < 1e-9
    assert abs(r["decay_step"][1] / r["decay_step"][0] - 10.0) < 1e-9


def test_grxvi_variance_within_three_sigma_of_target():
    fan_in = fan_out = 60
    r = geron_glorot_xavier_init(fan_in, fan_out, "normal", seed=7)
    n = fan_in * fan_out
    target = 2.0 / (fan_in + fan_out)
    assert abs(r["target_variance"] - target) < 1e-15
    # sd of a sample variance of n normals is target*sqrt(2/n)
    tol = 3 * target * math.sqrt(2.0 / n)
    assert abs(r["achieved_variance"] - target) < tol


def test_grxvi_uniform_limit_and_bounds():
    r = geron_glorot_xavier_init(4, 6, "uniform", seed=3)
    limit = math.sqrt(6.0 / 10)
    assert abs(r["scale"] - limit) < TOL
    W = np.array(r["weights"])
    assert W.shape == (4, 6)
    assert np.all(np.abs(W) <= limit)


# ------------------------------------------------------------ quantization


def test_grq8_scale_and_round_trip_error():
    x = [1.0, 0.5, -1.0, 0.0, 0.123]
    r = geron_int8_quantization(x)
    s = max(abs(v) for v in x) / 127
    assert abs(r["scale"] - s) < 1e-15
    assert r["q"] == [float(np.round(v / s)) for v in x]
    assert r["max_abs_error"] <= s / 2 + 1e-12
    assert r["dequantized"][3] == 0.0


def test_grq8_codes_stay_inside_the_symmetric_range():
    q = geron_int8_quantization(lcg(50, seed=9))["q"]
    assert max(q) <= 127 and min(q) >= -127


def test_grqat_straight_through_mask_and_clipping():
    r = geron_quantization_aware_training([0.24, 20.0], s=0.1, upstream_grad=[3.0, 3.0])
    assert np.allclose(r["y"], [0.2, 12.7], atol=1e-12)
    assert r["ste_mask"] == [1.0, 0.0]
    assert r["grad_x"] == [3.0, 0.0]
    assert r["clipped_fraction"] == 0.5


def test_grqat_is_transparent_in_range():
    x = [0.3, -0.7]
    r = geron_quantization_aware_training(x, s=0.1)
    assert r["grad_x"] == [1.0, 1.0]
    assert np.allclose(r["y"], [round(v / 0.1) * 0.1 for v in x], atol=1e-12)


def test_grptq_calibrates_each_layer_range():
    r = geron_static_ptq([lambda a: 2 * a, lambda a: a + 1], [[1.0], [2.0]])
    assert np.allclose(r["activation_ranges"], [2.0, 4.0, 5.0], atol=TOL)
    assert np.allclose(r["scales"], [v / 127 for v in (2.0, 4.0, 5.0)], atol=1e-15)


def test_grptq_enforces_the_layer_contract():
    with pytest.raises(ValueError):
        geron_static_ptq([lambda a: a[:1]], [[1.0], [2.0]])   # batch size changed
    with pytest.raises(ValueError):
        geron_static_ptq([], [[1.0]])


def test_grprn_prunes_exactly_the_smallest_magnitudes():
    W = [[1.0, -0.1], [0.05, 2.0]]
    r = geron_weight_pruning(W, 0.5)
    assert r["W_pruned"] == [[1.0, 0.0], [0.0, 2.0]]
    assert r["achieved_sparsity"] == 0.5
    assert r["n_pruned"] == 2


def test_grprn_zero_sparsity_is_a_no_op():
    W = [[-3.0, 0.2]]
    assert geron_weight_pruning(W, 0.0)["W_pruned"] == W


# ------------------------------------------------------- sampling / decoding


def test_grtmp_temperature_monotone_in_entropy():
    z = [2.0, 1.0, 0.0]
    ents = [geron_temperature_sampling(z, T=t)["entropy"] for t in (0.5, 1.0, 4.0)]
    assert ents[0] < ents[1] < ents[2]
    d = sum(math.exp(v / 0.5) for v in z)
    p0 = math.exp(z[0] / 0.5) / d
    assert abs(geron_temperature_sampling(z, T=0.5)["probabilities"][0] - p0) < TOL


def test_grtmp_rejects_zero_temperature():
    with pytest.raises(ValueError):
        geron_temperature_sampling([1.0, 2.0], T=0.0)


def test_grtop_renormalises_over_the_kept_set():
    z = [2.0, 1.0, 0.0]
    full = geron_temperature_sampling(z)["probabilities"]
    r = geron_topk_sampling(z, k=2)
    mass = full[0] + full[1]
    assert abs(r["kept_mass"] - mass) < TOL
    assert abs(r["probabilities"][0] - full[0] / mass) < TOL
    assert r["probabilities"][2] == 0.0
    assert abs(sum(r["probabilities"]) - 1.0) < 1e-12


def test_grtop_k_equals_one_is_greedy():
    assert geron_topk_sampling([2.0, 1.0, 0.0], k=1)["probabilities"] == [1.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        geron_topk_sampling([2.0, 1.0], k=5)


# ------------------------------------------------------------ ensembles


def test_grvoth_majority_and_tie_rule():
    r = geron_hard_voting([[1, 0], [1, 1], [0, 0]])
    assert r["y_hat"] == [1, 0]
    assert r["vote_counts"] == [[1, 2], [2, 1]]
    t = geron_hard_voting([[0], [1]])
    assert t["y_hat"] == [0] and t["ties"] == 1


def test_grvots_beats_hard_voting_on_confidence():
    P = [[[0.55, 0.45]], [[0.55, 0.45]], [[0.05, 0.95]]]
    soft = geron_soft_voting(P)
    hard = geron_hard_voting([[0], [0], [1]])
    assert soft["y_hat"] == [1] and hard["y_hat"] == [0]
    assert abs(soft["mean_probabilities"][0][1] - (0.45 + 0.45 + 0.95) / 3) < TOL


def test_grvots_rejects_non_distributions():
    with pytest.raises(ValueError):
        geron_soft_voting([[[0.5, 0.2]]])


def test_grstk_blender_recovers_truth_from_biased_bases():
    P = [[1.5, 0.5], [2.5, 1.5], [3.5, 2.5]]
    y = [1.0, 2.0, 3.0]
    r = geron_stacking_predictor(P, y)
    assert np.allclose(r["predictions"], y, atol=1e-10)
    assert r["rmse"] < min(r["base_rmse"])


def test_grstk_enforces_custom_blender_contract():
    with pytest.raises(ValueError):
        geron_stacking_predictor([[1.0]], [1.0], blender=lambda P, y: 42)


def test_grxgbg_gain_matches_the_formula():
    GL, HL, GR, HR, lam, gamma = -2.0, 2.0, 2.0, 2.0, 1.0, 0.5
    want = 0.5 * (
        GL**2 / (HL + lam) + GR**2 / (HR + lam) - (GL + GR) ** 2 / (HL + HR + lam)
    ) - gamma
    r = geron_xgboost_gain(GL, HL, GR, HR, lam, gamma)
    assert abs(r["gain"] - want) < TOL
    assert abs(r["left_weight"] - (-GL / (HL + lam))) < TOL


def test_grxgbg_gamma_prunes():
    a = geron_xgboost_gain(-2.0, 2.0, 2.0, 2.0, 1.0, 0.0)
    b = geron_xgboost_gain(-2.0, 2.0, 2.0, 2.0, 1.0, 5.0)
    assert a["should_split"] and not b["should_split"]
    assert abs((a["gain"] - b["gain"]) - 5.0) < TOL


def test_grtrc_majority_and_gini():
    y = [1] * 49 + [2] * 5
    r = geron_tree_classification_leaf(y)
    p = [0.0, 49 / 54, 5 / 54]
    assert r["prediction"] == 1
    assert abs(r["gini"] - (1 - sum(v * v for v in p))) < TOL


def test_grtrv_leaf_is_the_mean_and_mse_is_its_variance():
    y = [1.0, 2.0, 6.0]
    r = geron_tree_regression_leaf(y)
    assert abs(r["prediction"] - 3.0) < TOL
    assert abs(r["mse"] - float(np.var(y))) < TOL


def test_grtrv_empty_leaf_raises():
    with pytest.raises(ValueError):
        geron_tree_regression_leaf([1.0, 2.0], [False, False])


def test_grovr_trains_k_classifiers_and_separates_clusters():
    X = [[0.0], [0.5], [5.0], [5.5], [10.0], [10.5]]
    y = [0, 0, 1, 1, 2, 2]
    r = geron_one_vs_rest(X, y)
    assert r["n_classifiers"] == 3
    assert r["predictions"] == y and r["accuracy"] == 1.0


def test_grovo_uses_k_choose_two_duels():
    X = [[0.0], [0.5], [5.0], [5.5], [10.0], [10.5]]
    y = [0, 0, 1, 1, 2, 2]
    r = geron_one_vs_one(X, y)
    assert r["n_classifiers"] == 3 * 2 // 2
    assert r["predictions"] == y
    # every one of the 3 duels casts exactly one vote per instance
    assert [sum(v) for v in r["votes"]] == [3] * 6


# ----------------------------------------------------- unsupervised


def test_grsil_matches_hand_computed_distances():
    X = [[0.0], [1.0], [10.0], [11.0]]
    lab = [0, 0, 1, 1]
    a = [1.0, 1.0, 1.0, 1.0]
    b = [(10 + 11) / 2, (9 + 10) / 2, (9 + 10) / 2, (10 + 11) / 2]
    want = float(np.mean([(bb - aa) / max(aa, bb) for aa, bb in zip(a, b)]))
    r = geron_silhouette_score(X, lab)
    assert abs(r["silhouette"] - want) < TOL
    assert np.allclose(r["b"], b, atol=TOL)


def test_grsil_needs_two_clusters():
    with pytest.raises(ValueError):
        geron_silhouette_score([[0.0], [1.0]], [0, 0])


def test_grpcap_recovers_a_one_dimensional_subspace():
    X = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]
    r = geron_pca_projection(X, 1)
    assert abs(r["explained_variance_ratio"][0] - 1.0) < 1e-12
    assert np.allclose(np.abs(r["components"][0]), [1 / math.sqrt(2)] * 2, atol=TOL)
    assert np.allclose([v[0] for v in r["projection"]], [-math.sqrt(2), 0.0, math.sqrt(2)], atol=TOL)


def test_grpcap_variance_ratios_match_eigenvalues():
    u = lcg(24, seed=13)
    X = np.array(u).reshape(12, 2)
    Xc = X - X.mean(axis=0)
    ev = np.sort(np.linalg.eigvalsh(Xc.T @ Xc))[::-1]
    r = geron_pca_projection(X, 2)
    assert np.allclose(r["explained_variance_ratio"], ev / ev.sum(), atol=1e-10)


def test_grnmfo_objective_is_the_frobenius_norm():
    X = [[1.0, 2.0], [2.0, 4.0]]
    W, H = [[1.0], [2.0]], [[0.5, 1.0]]
    R = np.array(W) @ np.array(H)
    want = float(np.sum((np.array(X) - R) ** 2))
    assert abs(geron_nmf_objective(X, W, H)["objective"] - want) < TOL


def test_grnmfo_rejects_negative_factors():
    with pytest.raises(ValueError):
        geron_nmf_objective([[1.0]], [[-1.0]], [[1.0]])


def test_grwpc_score_is_the_independence_ratio():
    counts = {"h": 15, "u": 20, "g": 4, "s": 5}
    pairs = {("h", "u"): 10, ("g", "s"): 4}
    r = geron_wordpiece_tokenizer_score(counts, pairs)
    assert abs(r["scores"][("h", "u")] - 10 / (15 * 20)) < TOL
    assert r["best_pair"] == ("g", "s")


def test_grwpc_rejects_impossible_pair_counts():
    with pytest.raises(ValueError):
        geron_wordpiece_tokenizer_score({"a": 2, "b": 3}, {("a", "b"): 5})


# ------------------------------------------------------------------ RL


def test_grret_backward_recursion_matches_explicit_sum():
    rew = [10.0, 0.0, -50.0, 4.0]
    g = 0.8
    want = [sum(g**k * rew[t + k] for k in range(len(rew) - t)) for t in range(len(rew))]
    got = geron_discounted_return(rew, g)["returns"]
    assert np.allclose(got, want, atol=1e-12)
    assert abs(got[0] - float(np.mean(rew))) > 1.0        # not a mean-of-inputs stub


def test_grret_horizon_and_gamma_zero():
    assert geron_discounted_return([1.0, 2.0], 0.0)["returns"] == [1.0, 2.0]
    assert abs(geron_discounted_return([1.0], 0.9)["effective_horizon"] - 10.0) < TOL


def test_grvpi_matches_the_geometric_series():
    P = [[[1.0]]]
    r = geron_state_value_function(0, [0], P, [[1.0]], gamma=0.5)
    assert abs(r["value"] - 1 / (1 - 0.5)) < 1e-12


def test_grvpi_two_state_cycle_solved_exactly():
    P = [[[0.0, 1.0]], [[1.0, 0.0]]]
    R = [[1.0], [0.0]]
    g = 0.9
    V = geron_state_value_function(0, [0, 0], P, R, g)["values"]
    # V0 = 1 + g V1 and V1 = g V0
    assert abs(V[0] - 1 / (1 - g * g)) < 1e-10
    assert abs(V[1] - g * V[0]) < 1e-10


def test_grqpi_is_one_bellman_backup_of_v():
    P = [[[1.0], [1.0]]]
    R = [[1.0, 0.0]]
    g = 0.5
    r = geron_action_value_function(0, 1, [0], P, R, g)
    V = r["values"][0]
    assert abs(r["q_value"] - (0.0 + g * V)) < 1e-12
    assert abs(r["advantage"] - (r["q_value"] - V)) < 1e-12


def test_grql_update_and_terminal_handling():
    Q = [[0.0, 0.0], [10.0, 5.0]]
    r = geron_q_learning_update(Q, 0, 0, 1.0, 1, alpha=0.1, gamma=0.9)
    assert abs(r["target"] - (1.0 + 0.9 * 10.0)) < TOL
    assert abs(r["new_value"] - (0.0 + 0.1 * (10.0 - 0.0))) < TOL
    assert geron_q_learning_update(Q, 0, 0, 1.0, 1, 0.1, 0.9, done=True)["target"] == 1.0


def test_grql_leaves_other_entries_untouched():
    Q = [[0.0, 7.0], [10.0, 5.0]]
    out = geron_q_learning_update(Q, 0, 0, 1.0, 1, 0.1, 0.9)["Q"]
    assert out[0][1] == 7.0 and out[1] == [10.0, 5.0]


def test_grtd0_update_matches_hand_arithmetic():
    r = geron_td_zero_update([0.0, 5.0], 0, 1, 1.0, alpha=0.2, gamma=0.9)
    assert abs(r["target"] - (1.0 + 0.9 * 5.0)) < TOL
    assert abs(r["new_value"] - 0.2 * 5.5) < TOL
    assert geron_td_zero_update([5.5, 5.0], 0, 1, 1.0, 0.2, 0.9)["td_error"] == 0.0


def test_grrein_gradient_is_the_return_weighted_score_sum():
    S = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    G = [2.0, -1.0, 0.5]
    want = np.array(S).T @ np.array(G)
    r = geron_reinforce_policy_gradient([0.0, 0.0], S, G, alpha=0.5)
    assert np.allclose(r["gradient"], want, atol=TOL)
    assert np.allclose(r["theta_new"], 0.5 * want, atol=TOL)


def test_grrein_mean_baseline_centres_advantages():
    r = geron_reinforce_policy_gradient(
        [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [2.0, -1.0], 0.5, baseline="mean"
    )
    assert abs(sum(r["advantages"])) < 1e-12
    with pytest.raises(ValueError):
        geron_reinforce_policy_gradient([0.0], [-1.0, -2.0], [1.0, 1.0], 0.5)


def test_grppo_min_branch_not_just_the_clip():
    r = geron_ppo_clipped_objective([1.5, 0.5], [1.0, -1.0], eps=0.2)
    assert np.allclose(r["per_step"], [min(1.5, 1.2) * 1.0, min(0.5 * -1, 0.8 * -1)], atol=TOL)
    assert r["per_step"] == [1.2, -0.8]


def test_grppo_inside_trust_region_is_unclipped():
    r = geron_ppo_clipped_objective([1.0, 1.1], [3.0, -2.0], eps=0.2)
    assert r["clipped_fraction"] == 0.0
    assert np.allclose(r["per_step"], [3.0, 1.1 * -2.0], atol=TOL)
    with pytest.raises(ValueError):
        geron_ppo_clipped_objective([-0.5], [1.0])


def test_grpex_weights_follow_the_is_formula():
    d = [3.0, 1.0]
    alpha, beta = 1.0, 1.0
    P = [v / sum(d) for v in d]
    N = 2
    w = [(N * p) ** (-beta) for p in P]
    w = [v / max(w) for v in w]
    r = geron_prioritized_experience_weight(d, alpha=alpha, beta=beta, eps=0.0)
    assert np.allclose(r["probabilities"], P, atol=TOL)
    assert np.allclose(r["weights"], w, atol=TOL)
    assert max(r["weights"]) == 1.0


def test_grpex_alpha_zero_is_uniform():
    r = geron_prioritized_experience_weight([3.0, 1.0, 8.0], alpha=0.0, beta=1.0)
    assert np.allclose(r["probabilities"], [1 / 3] * 3, atol=TOL)
    assert np.allclose(r["weights"], [1.0] * 3, atol=TOL)


def test_grrlhf_objective_is_reward_minus_beta_kl():
    r = [1.0, 2.0]
    lp = [-1.0, -0.5]
    lr = [-2.0, -2.5]
    beta = 0.5
    kl = float(np.mean([a - b for a, b in zip(lp, lr)]))
    want = float(np.mean(r)) - beta * kl
    got = geron_rlhf_reward_kl_objective(r, lp, lr, beta)
    assert abs(got["objective"] - want) < TOL
    assert abs(got["kl"] - kl) < TOL


def test_grrlhf_identical_policy_pays_no_penalty():
    r = geron_rlhf_reward_kl_objective([1.0, 3.0], [-1.0, -2.0], [-1.0, -2.0], 0.9)
    assert r["kl"] == 0.0 and abs(r["objective"] - 2.0) < TOL


# ------------------------------------------------ autoencoders / generative


def test_grsae_loss_is_reconstruction_plus_l1():
    x = [[1.0, 0.0]]
    h = [[0.0, 2.0, -1.0]]
    dec = [[0.5, 0.0]]
    lam = 0.5
    want = 0.25 + lam * 3.0
    r = geron_sparse_autoencoder(x, h, dec, lam)
    assert abs(r["loss"] - want) < TOL
    assert abs(r["sparsity"] - 1 / 3) < TOL


def test_grstae_tied_decoder_is_the_transpose():
    W = [[1.0], [1.0]]
    r = geron_stacked_autoencoder([[1.0, 2.0]], [W], activation="linear")
    assert r["code"] == [[3.0]]
    assert r["reconstruction"] == [[3.0, 3.0]]
    assert abs(r["reconstruction_error"] - float(np.mean([(3 - 1) ** 2, (3 - 2) ** 2]))) < TOL


def test_grstae_requires_a_bottleneck():
    with pytest.raises(ValueError):
        geron_stacked_autoencoder([[1.0, 2.0]], [[[1.0, 0.0], [0.0, 1.0]]])


def test_grvae_kl_matches_the_closed_form():
    mu = np.array([[1.0, -0.5]])
    lv = np.array([[0.0, 0.3]])
    kl = float(-0.5 * np.sum(1 + lv - mu**2 - np.exp(lv)))
    r = geron_vae_elbo([[0.0, 0.0]], mu, lv, [[0.0, 0.0]])
    assert abs(r["kl"] - kl) < TOL
    assert abs(r["elbo"] - (0.0 - kl)) < TOL
    assert abs(r["loss"] + r["elbo"]) < TOL


def test_grvae_bernoulli_term_is_a_log_sum():
    r = geron_vae_elbo([[1.0, 0.0]], [[0.0]], [[0.0]], [[0.8, 0.3]], likelihood="bernoulli")
    want = math.log(0.8) + math.log(0.7)
    assert abs(r["reconstruction_term"] - want) < 1e-9


def test_grrep_uses_half_the_logvar():
    r = geron_reparameterization_trick([0.0], [2 * math.log(3)], eps=[1.0])
    assert abs(r["sigma"][0] - 3.0) < 1e-12
    assert abs(r["z"][0] - 3.0) < 1e-12
    assert abs(r["dz_dlogvar"][0] - 0.5 * 3.0 * 1.0) < 1e-12


def test_grrep_lcg_noise_is_reproducible_and_roughly_standard():
    a = geron_reparameterization_trick(np.zeros(400), np.zeros(400), seed=3)
    b = geron_reparameterization_trick(np.zeros(400), np.zeros(400), seed=3)
    assert a["z"] == b["z"]
    # sample variance of 400 standard normals: sd = sqrt(2/400) = 0.0707
    assert abs(a["sample_variance"] - 1.0) < 3 * math.sqrt(2 / 400)
    assert abs(a["sample_mean"]) < 3 / math.sqrt(400)


def test_grscm_loss_is_the_squared_target_residual():
    x0 = [[0.0, 0.0]]
    eps = [[1.0, -2.0]]
    sigma = 0.5
    pred = [[0.0, 0.0]]
    target = [1.0 / sigma, -2.0 / sigma]
    want = sum(v * v for v in target)
    assert abs(geron_score_matching_loss(x0, sigma, eps, pred)["loss"] - want) < TOL


def test_grscm_sigma2_weighting_rescales():
    x0, eps, pred = [[0.0]], [[1.0]], [[0.0]]
    plain = geron_score_matching_loss(x0, 0.5, eps, pred)["loss"]
    weighted = geron_score_matching_loss(x0, 0.5, eps, pred, weight="sigma2")["loss"]
    assert abs(weighted - 0.25 * plain) < TOL


# ------------------------------------------------------------- NLP heads


def test_grsnt_probability_is_sigmoid_of_pooled_score():
    E = [[1.0], [2.0]]
    r = geron_sentiment_binary([0, 1], E, [1.0], b=0.5)
    logit = (1.0 + 2.0) / 2 + 0.5
    assert abs(r["logit"] - logit) < TOL
    assert abs(r["probability"] - 1 / (1 + math.exp(-logit))) < TOL


def test_grsnt_max_pooling_differs_from_mean():
    E = [[1.0], [2.0]]
    assert geron_sentiment_binary([0, 1], E, [1.0], pooling="max")["pooled"] == [2.0]
    with pytest.raises(ValueError):
        geron_sentiment_binary([0, 5], E, [1.0])


def test_grsft_averages_over_response_tokens_only():
    logits = [[10.0, 0.0], [0.0, 0.0], [1.0, 3.0]]
    mask = [False, True, True]
    tgt = [0, 1, 1]
    lse2 = math.log(math.exp(1.0) + math.exp(3.0))
    want = (math.log(2.0) + (lse2 - 3.0)) / 2
    r = geron_sft_objective(logits, mask, tgt)
    assert abs(r["loss"] - want) < 1e-12
    assert r["n_response_tokens"] == 2


def test_grsft_all_masked_raises():
    with pytest.raises(ValueError):
        geron_sft_objective([[1.0, 0.0]], [False], [0])


def test_grnsp_loss_matches_hand_log_softmax():
    r = geron_bert_nsp_loss([[0.0, 5.0], [2.0, 1.0]], [1, 0])
    l0 = math.log(math.exp(0.0) + math.exp(5.0)) - 5.0
    l1 = math.log(math.exp(2.0) + math.exp(1.0)) - 2.0
    assert abs(r["loss"] - (l0 + l1) / 2) < 1e-12
    assert abs(r["baseline_loss"] - math.log(2)) < TOL


def test_grnsp_rejects_non_binary_labels():
    with pytest.raises(ValueError):
        geron_bert_nsp_loss([[0.0, 1.0]], [2])


# --------------------------------------------------------- autodiff / misc


def test_grnud_central_difference_is_exact_for_quadratics():
    r = geron_numerical_differentiation(lambda t: 3 * t**2 + 2 * t, 4.0)
    assert abs(r["derivative"] - (6 * 4.0 + 2)) < 1e-6
    assert r["step_error"] < 1e-6


def test_grnud_gradient_of_a_multivariate_function():
    g = geron_numerical_differentiation(lambda v: v[0] ** 2 + 3 * v[1], [1.0, 1.0])
    assert np.allclose(g["derivative"], [2.0, 3.0], atol=1e-6)
    with pytest.raises(ValueError):
        geron_numerical_differentiation(lambda t: t, 1.0, h=0.0)


def test_grrad_accumulates_fan_out():
    g = {"L": {"u": 1.0, "v": 1.0}, "u": {"x": 2.0}, "v": {"x": 3.0}}
    assert geron_reverse_mode_autodiff(g)["gradients"]["x"] == 5.0


def test_grrad_matches_a_hand_chain_rule_and_rejects_cycles():
    # L = 2*y, y = 3*x  ->  dL/dx = 6
    g = {"L": {"y": 2.0}, "y": {"x": 3.0}}
    assert geron_reverse_mode_autodiff(g)["leaf_gradients"] == {"x": 6.0}
    with pytest.raises(ValueError):
        geron_reverse_mode_autodiff({"a": {"b": 1.0}, "b": {"a": 1.0}}, output="a")


def test_grsmd_derivative_agrees_with_numerical_differentiation():
    f = ("+", ("*", ("^", "x", 3), 2.0), ("sin", "x"))
    d = geron_symbolic_differentiation(f, "x", at={"x": 1.3})
    num = geron_numerical_differentiation(
        lambda t: 2.0 * t**3 + math.sin(t), 1.3, h=1e-5
    )["derivative"]
    assert abs(d["value"] - num) < 1e-6


def test_grsmd_product_and_quotient_rules():
    q = geron_symbolic_differentiation(("/", "x", ("+", "x", 1.0)), "x", at={"x": 2.0})
    assert abs(q["value"] - 1.0 / (2.0 + 1.0) ** 2) < 1e-9
    with pytest.raises(ValueError):
        geron_symbolic_differentiation(("bogus", "x"), "x")


def test_grrnd_finds_the_best_sampled_configuration():
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0.0, 1.0, 2.0, 3.0]
    scorer = lambda Xtr, ytr, Xva, yva, p: -abs(p["alpha"] - 2.0)  # noqa: E731
    r = geron_randomized_search_cv(X, y, {"alpha": (0.0, 10.0)}, 8, 2, fit_score=scorer, seed=4)
    assert len(r["results"]) == 8
    assert r["best_score"] == max(row["mean_score"] for row in r["results"])
    assert abs(r["best_score"] + abs(r["best_params"]["alpha"] - 2.0)) < 1e-12


def test_grrnd_requires_fit_score_and_valid_k():
    X, y = [[0.0], [1.0]], [0.0, 1.0]
    with pytest.raises(ValueError):
        geron_randomized_search_cv(X, y, {"a": (0, 1)}, 2, 2)
    with pytest.raises(ValueError):
        geron_randomized_search_cv(
            X, y, {"a": (0, 1)}, 2, 9, fit_score=lambda *a: 0.0
        )


def test_grrsk_adds_the_identity_shortcut():
    r = geron_resnet_skip([1.0, 2.0], [0.5, -1.0])
    assert r["output"] == [1.5, 1.0]
    assert geron_resnet_skip([1.0, 2.0], [0.0, 0.0])["output"] == [1.0, 2.0]


def test_grrsk_shape_change_needs_a_projection():
    with pytest.raises(ValueError):
        geron_resnet_skip([1.0], [0.0, 0.0])
    r = geron_resnet_skip([1.0], [0.0, 0.0], projection=[[2.0, 3.0]])
    assert r["output"] == [2.0, 3.0]


def test_gryol_loss_terms_are_separately_correct():
    t = [[[0.5, 0.5, 0.25, 0.25, 1.0, 1.0]]]
    p = [[[0.6, 0.5, 0.25, 0.25, 1.0, 1.0]]]
    r = geron_yolo_grid_loss(p, t)
    assert abs(r["loss_coord"] - 5.0 * 0.1**2) < 1e-12
    assert r["loss_obj"] == 0.0 and r["loss_class"] == 0.0


def test_gryol_sqrt_wh_and_noobj_weighting():
    t = [[[0.5, 0.5, 0.25, 0.0, 1.0, 1.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    p = [[[0.5, 0.5, 1.0, 0.0, 1.0, 1.0]], [[0.9, 0.9, 0.5, 0.5, 0.2, 0.7]]]
    r = geron_yolo_grid_loss(p, t)
    assert abs(r["loss_coord"] - 5.0 * (math.sqrt(1.0) - math.sqrt(0.25)) ** 2) < 1e-12
    assert abs(r["loss_noobj"] - 0.5 * 0.2**2) < 1e-12
    assert r["n_objects"] == 1


# ------------------------------------------------- shared payload contract


ALL_CALLS = [
    (geron_sigmoid, ([0.0, 1.0],), {}),
    (geron_tanh_activation, ([0.0, 1.0],), {}),
    (geron_rmse, ([1.0, 2.0], [1.5, 2.5]), {}),
    (geron_discounted_return, ([1.0, 2.0], 0.5), {}),
    (geron_xgboost_gain, (-2.0, 2.0, 2.0, 2.0), {}),
    (geron_int8_quantization, ([1.0, -0.5],), {}),
    (geron_tree_regression_leaf, ([1.0, 3.0],), {}),
]


@pytest.mark.parametrize("fn,args,kwargs", ALL_CALLS)
def test_payload_has_the_house_keys(fn, args, kwargs):
    r = fn(*args, **kwargs)
    for key in ("estimate", "n", "method"):
        assert key in r, f"{fn.__name__} payload is missing {key!r}"
    assert isinstance(r["method"], str) and r["method"]


def test_no_module_returns_a_mean_of_its_inputs():
    """The placeholder bodies all returned float(np.mean(first_arg))."""
    checks = [
        (geron_rmse([1.0, 2.0, 9.0], [1.0, 2.0, 3.0])["estimate"], float(np.mean([1.0, 2.0, 9.0]))),
        (geron_discounted_return([10.0, 0.0, -50.0], 0.8)["estimate"], float(np.mean([10.0, 0.0, -50.0]))),
        (geron_tree_classification_leaf([1, 1, 2])["estimate"], float(np.mean([1, 1, 2]))),
        (geron_ppo_clipped_objective([1.5, 0.5], [1.0, -1.0])["estimate"], float(np.mean([1.5, 0.5]))),
        (geron_nmf_objective([[1.0, 2.0]], [[1.0]], [[0.5, 1.0]])["estimate"], 1.5),
    ]
    for got, stub in checks:
        assert abs(got - stub) > 1e-6
