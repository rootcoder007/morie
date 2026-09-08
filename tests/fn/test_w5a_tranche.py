"""Independent-route tests for the w5a tranche (Geron training mechanics + DL).

Every assertion is checked against a route the implementation does not
take: hand-derived optimizer steps, closed-form schedules, finite
differences, log-sums computed with ``math``, and property checks
(norms, distributions, monotonicity).  Nothing here calls the module
under test twice and compares it with itself.

Deterministic data only -- explicit lists, or the reference LCG
``s = (1664525 s + 1013904223) % 2**32; u = (s + 0.5) / 2**32``.
"""

import math

from morie.fn import _array_core as np
import pytest

from morie.fn.gredsq import geron_encoder_decoder_seq2seq
from morie.fn.greast import geron_early_stopping
from morie.fn.grdino import geron_dino_self_distillation
from morie.fn.grdlm import geron_dataloader_minibatch
from morie.fn.grdpml import geron_ddpm_simple_loss
from morie.fn.grdpmf import geron_ddpm_forward_process
from morie.fn.grdpmr import geron_ddpm_reverse_step
from morie.fn.grdpo import geron_dpo_loss
from morie.fn.grdqnl import geron_dqn_loss
from morie.fn.grdro import geron_dropout
from morie.fn.grduel import geron_dueling_dqn
from morie.fn.grdyq import geron_dynamic_quantization
from morie.fn.grelas import geron_elastic_net_cost
from morie.fn.gremb import geron_embedding_lookup
from morie.fn.grent import geron_shannon_entropy
from morie.fn.grepl import geron_epsilon_greedy
from morie.fn.grevr import geron_explained_variance_ratio
from morie.fn.grf1 import geron_f1_score
from morie.fn.grfad import Dual, geron_forward_mode_autodiff
from morie.fn.grfcn import geron_fcn_upsample
from morie.fn.grffn import geron_transformer_feedforward
from morie.fn.grfim import geron_feature_importance_mdi
from morie.fn.grflam import geron_flamingo_cross_modal_attn
from morie.fn.grflash import geron_flash_attention_tile
from morie.fn.grfmp import geron_feature_map_dim
from morie.fn.grfp6 import geron_fp16_mixed_precision
from morie.fn.grgan import geron_gan_minimax
from morie.fn.grgbm import geron_gradient_boosting_residual
from morie.fn.grgcl import geron_gradient_clipping
from morie.fn.grgin import geron_gini_impurity
from morie.fn.grgmem import geron_gmm_em_step
from morie.fn.grgmll import geron_gmm_log_likelihood
from morie.fn.grgptl import geron_gpt_autoregressive_loss
from morie.fn.grgrp import geron_gaussian_random_projection
from morie.fn.grgruc import geron_gru_cell
from morie.fn.grgs import geron_grid_search_cv
from morie.fn.grhbb import geron_hebb_rule
from morie.fn.grhei import geron_he_init
from morie.fn.grhev import geron_heaviside_step
from morie.fn.grig import geron_information_gain
from morie.fn.grimp import geron_simple_imputer
from morie.fn.grinc import geron_in_context_learning
from morie.fn.grjll import geron_johnson_lindenstrauss_bound
from morie.fn.grkdl import geron_knowledge_distillation_loss
from morie.fn.grkfd import geron_kfold_cv
from morie.fn.grkldg import geron_kl_divergence_gaussian
from morie.fn.grkmo import geron_kmeans_objective
from morie.fn.grkmpp import geron_kmeans_pp_seeding
from morie.fn.grkpc import geron_kernel_pca_rbf
from morie.fn.grkvc import geron_kv_cache_compression
from morie.fn.grlaso import geron_lasso_cost
from morie.fn.grlinf import geron_linear_layer_forward
from morie.fn.grln import geron_layer_normalization
from morie.fn.grlof import geron_local_outlier_factor
from morie.fn.grlogc import geron_logistic_cross_entropy_cost
from morie.fn.grlogg import geron_logistic_cost_gradient
from morie.fn.grlogp import geron_logistic_regression_probability
from morie.fn.grlrco import geron_lr_cosine_annealing
from morie.fn.grlrex import geron_lr_exponential_schedule
from morie.fn.grlrnc import geron_learning_curves
from morie.fn.grlstc import geron_lstm_cell
from morie.fn.grmae import geron_mae
from morie.fn.grmcol import geron_gan_mode_collapse_metric
from morie.fn.grmgd import geron_minibatch_gradient_descent
from morie.fn.grmha import geron_multi_head_attention
from morie.fn.grmlb import geron_multilabel_classification
from morie.fn.grmlc import geron_classification_mlp_output
from morie.fn.grmlm import geron_bert_mlm_loss
from morie.fn.grmlpf import geron_mlp_forward
from morie.fn.grmlr import geron_regression_mlp_output
from morie.fn.grmms import geron_minmax_scaler
from morie.fn.grmnr import geron_max_norm_regularization
from morie.fn.grmom import geron_momentum_update
from morie.fn.grmpl import geron_max_pooling
from morie.fn.grmse import geron_linreg_mse_cost
from morie.fn.grn001 import geron_ch4_simple_linear_life_satisfaction
from morie.fn.grn002 import geron_ch4_linear_regression_prediction
from morie.fn.grn005 import geron_ch4_normal_equation
from morie.fn.grn007 import geron_ch4_mse_gradient_vector
from morie.fn.grn011 import geron_ch4_lasso_regression_cost_function
from morie.fn.grn013 import geron_ch4_elastic_net_cost_function


# --------------------------------------------------------------------------
# deterministic data
# --------------------------------------------------------------------------

def lcg(n, seed=0):
    """The reference LCG, so test data is reproducible without numpy.random."""
    s = seed % 2**32
    out = []
    for _ in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out.append((s + 0.5) / 2**32)
    return out


X_LINE = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]
Y_LINE = [1.0, 2.0, 3.0, 4.0]


# --------------------------------------------------------------------------
# Ch 4: linear regression family
# --------------------------------------------------------------------------

def test_grmse_matches_hand_sum_of_squares():
    theta = [0.5, 0.8]
    hand = sum((0.5 + 0.8 * x[1] - y) ** 2 for x, y in zip(X_LINE, Y_LINE)) / 4
    assert geron_linreg_mse_cost(X_LINE, Y_LINE, theta)["cost"] == pytest.approx(hand)


def test_grmse_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        geron_linreg_mse_cost(X_LINE, Y_LINE, [1.0])


def test_grmse_is_not_a_mean_of_inputs_stub():
    r = geron_linreg_mse_cost(X_LINE, Y_LINE, [0.0, 1.0])
    assert r["cost"] == pytest.approx(0.0)
    assert r["cost"] != pytest.approx(np.mean(Y_LINE))


def test_grn002_matches_explicit_weighted_sum():
    r = geron_ch4_linear_regression_prediction([1.0, 2.0, 3.0], [3.0, 4.0])
    assert r["prediction"] == pytest.approx(1.0 + 2 * 3 + 3 * 4)


def test_grn001_is_grn002_with_one_feature():
    r = geron_ch4_simple_linear_life_satisfaction(4.85, 4.91e-5, 27195.0)
    assert r["life_satisfaction"] == pytest.approx(4.85 + 4.91e-5 * 27195.0)


def test_grn005_matches_textbook_slope_intercept():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [2.0, 3.0, 5.0, 4.0]
    xb, yb = sum(xs) / 4, sum(ys) / 4
    b1 = sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / sum((x - xb) ** 2 for x in xs)
    b0 = yb - b1 * xb
    theta = geron_ch4_normal_equation([[1.0, x] for x in xs], ys)["theta"]
    assert theta == pytest.approx([b0, b1])


def test_grn005_rejects_collinear_columns():
    with pytest.raises(ValueError):
        geron_ch4_normal_equation([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], [1.0, 2.0, 3.0])


def test_grn007_matches_finite_difference_of_the_cost():
    theta = np.array([0.3, -0.2])
    g = geron_ch4_mse_gradient_vector(X_LINE, Y_LINE, theta)["gradient"]
    h = 1e-6
    fd = []
    for j in range(2):
        up, dn = theta.copy(), theta.copy()
        up[j] += h
        dn[j] -= h
        fd.append((geron_linreg_mse_cost(X_LINE, Y_LINE, up)["cost"]
                   - geron_linreg_mse_cost(X_LINE, Y_LINE, dn)["cost"]) / (2 * h))
    assert g == pytest.approx(fd, abs=1e-6)


def test_grlaso_penalty_excludes_the_intercept():
    r = geron_lasso_cost(X_LINE, Y_LINE, [3.0, 1.0], alpha=2.0)
    assert r["l1_penalty"] == pytest.approx(2.0)          # 2 * |1.0|, theta_0 skipped
    assert r["cost"] == pytest.approx(r["mse"] + 2.0)


def test_grn011_is_grlaso_at_double_alpha():
    a = geron_ch4_lasso_regression_cost_function(X_LINE, Y_LINE, [0.0, 1.0], alpha=0.7)
    b = geron_lasso_cost(X_LINE, Y_LINE, [0.0, 1.0], alpha=1.4)
    assert a["cost"] == pytest.approx(b["cost"])


def test_grelas_interpolates_lasso_and_ridge():
    th = [0.0, 2.0]
    lasso = geron_elastic_net_cost(X_LINE, Y_LINE, th, alpha=1.0, r=1.0)
    ridge = geron_elastic_net_cost(X_LINE, Y_LINE, th, alpha=1.0, r=0.0)
    assert lasso["l2_penalty"] == 0.0
    assert lasso["l1_penalty"] == pytest.approx(2.0)
    assert ridge["l1_penalty"] == 0.0
    assert ridge["l2_penalty"] == pytest.approx(0.5 * 4.0)


def test_grn013_ridge_arm_scales_as_one_over_m():
    th = [0.0, 1.0]
    small = geron_ch4_elastic_net_cost_function(X_LINE, Y_LINE, th, alpha=1.0, r=0.0)
    X8 = X_LINE + [[1.0, float(i)] for i in range(5, 9)]
    y8 = Y_LINE + [5.0, 6.0, 7.0, 8.0]
    big = geron_ch4_elastic_net_cost_function(X8, y8, th, alpha=1.0, r=0.0)
    assert small["l2_penalty"] == pytest.approx(1.0 / 4)
    assert big["l2_penalty"] == pytest.approx(1.0 / 8)


# --------------------------------------------------------------------------
# dense layers and MLPs
# --------------------------------------------------------------------------

def test_grlinf_matches_hand_dot_products():
    W = [[1.0, 2.0], [-1.0, 0.5]]
    r = geron_linear_layer_forward([3.0, 4.0], W, [1.0, -1.0])
    assert r["output"] == pytest.approx([1 * 3 + 2 * 4 + 1, -1 * 3 + 0.5 * 4 - 1])


def test_grlinf_rejects_wrong_width():
    with pytest.raises(ValueError):
        geron_linear_layer_forward([1.0, 2.0, 3.0], [[1.0, 2.0]], [0.0])


def test_grmlr_softplus_is_log1p_exp():
    r = geron_regression_mlp_output([2.0], [[1.5]], [-1.0], activation="softplus")
    assert r["prediction"][0] == pytest.approx(math.log1p(math.exp(1.5 * 2 - 1)))


def test_grmlc_softmax_matches_hand_exponentials():
    r = geron_classification_mlp_output([1.0], [[2.0], [0.0], [-1.0]], [0.0, 0.0, 0.0])
    e = [math.exp(2.0), 1.0, math.exp(-1.0)]
    tot = sum(e)
    assert r["probabilities"] == pytest.approx([v / tot for v in e])
    assert sum(r["probabilities"]) == pytest.approx(1.0)


def test_grmlpf_two_layers_by_hand():
    W = [[[1.0, 1.0], [1.0, -1.0]], [[2.0, 3.0]]]
    b = [[0.0, 0.0], [1.0]]
    r = geron_mlp_forward([1.0, 2.0], W, b)          # relu
    # layer1 pre-activation [3, -1] -> relu [3, 0]; layer2 2*3 + 3*0 + 1 = 7
    assert r["activations"][1] == pytest.approx([3.0, 0.0])
    assert r["output"] == pytest.approx([7.0])


def test_grffn_is_position_wise():
    W1 = [[1.0, -1.0], [0.0, 1.0]]
    W2 = [[1.0, 0.0], [0.0, 1.0]]
    rows = geron_transformer_feedforward([[1.0, 2.0], [5.0, -3.0]], W1, [0.0, 0.0],
                                         W2, [0.0, 0.0])["output"]
    one = geron_transformer_feedforward([1.0, 2.0], W1, [0.0, 0.0], W2, [0.0, 0.0])["output"]
    assert rows[0] == pytest.approx(one)


# --------------------------------------------------------------------------
# logistic regression
# --------------------------------------------------------------------------

def test_grlogp_matches_math_sigmoid():
    r = geron_logistic_regression_probability([[1.0, 2.0], [1.0, -3.0]], [0.5, 1.0])
    assert r["probability"] == pytest.approx([1 / (1 + math.exp(-2.5)),
                                              1 / (1 + math.exp(2.5))])


def test_grlogc_matches_hand_log_sum():
    X = [[1.0, 2.0], [1.0, -3.0]]
    th = [0.5, 1.0]
    p = [1 / (1 + math.exp(-2.5)), 1 / (1 + math.exp(2.5))]
    hand = -(math.log(p[0]) + math.log(1 - p[1])) / 2
    assert geron_logistic_cross_entropy_cost(X, [1.0, 0.0], th)["cost"] == pytest.approx(hand)


def test_grlogg_matches_finite_difference_of_log_loss():
    X = [[1.0, 2.0], [1.0, -3.0], [1.0, 0.5]]
    y = [1.0, 0.0, 1.0]
    th = np.array([0.2, -0.4])
    g = geron_logistic_cost_gradient(X, y, th)["gradient"]
    h = 1e-6
    fd = []
    for j in range(2):
        up, dn = th.copy(), th.copy()
        up[j] += h
        dn[j] -= h
        fd.append((geron_logistic_cross_entropy_cost(X, y, up)["cost"]
                   - geron_logistic_cross_entropy_cost(X, y, dn)["cost"]) / (2 * h))
    assert g == pytest.approx(fd, abs=1e-6)


def test_grlogc_rejects_non_binary_labels():
    with pytest.raises(ValueError):
        geron_logistic_cross_entropy_cost([[1.0]], [2.0], [0.0])


# --------------------------------------------------------------------------
# perceptron
# --------------------------------------------------------------------------

def test_grhev_is_closed_at_zero():
    assert geron_heaviside_step([0.0, -1e-12])["output"] == [1.0, 0.0]


def test_grhbb_update_is_an_outer_product():
    r = geron_hebb_rule([2.0, 0.0], [1.0, 0.0], [0.0, 1.0], [[0.0, 0.0], [0.0, 0.0]], eta=0.5)
    # error = [1, -1]; eta * outer([2, 0], [1, -1]) = [[1, -1], [0, 0]]
    assert np.asarray(r["delta_w"]) == pytest.approx(np.asarray([[1.0, -1.0], [0.0, 0.0]]))


def test_grhbb_makes_no_update_when_correct():
    r = geron_hebb_rule([2.0, 3.0], [1.0], [1.0], [[0.4], [0.9]], eta=0.5)
    assert np.asarray(r["w_new"]) == pytest.approx(np.asarray([[0.4], [0.9]]))
    assert r["converged"] is True


# --------------------------------------------------------------------------
# metrics and preprocessing
# --------------------------------------------------------------------------

def test_grmae_and_rmse_ordering():
    r = geron_mae([0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, 5.0])
    assert r["mae"] == pytest.approx((1 + 1 + 1 + 5) / 4)
    assert r["rmse"] == pytest.approx(math.sqrt((1 + 1 + 1 + 25) / 4))
    assert r["rmse"] >= r["mae"]


def test_grf1_matches_hand_counted_precision_recall():
    y = [1, 1, 1, 0, 0]
    p = [1, 1, 0, 1, 0]
    tp, fp, fn = 2, 1, 1
    assert geron_f1_score(y, p)["f1"] == pytest.approx(2 * tp / (2 * tp + fp + fn))


def test_grf1_is_below_the_arithmetic_mean_when_lopsided():
    r = geron_f1_score([1, 1, 1, 0], [1, 0, 0, 0])
    assert r["f1"] < (r["precision"] + r["recall"]) / 2


def test_grmms_maps_endpoints_exactly():
    r = geron_minmax_scaler([[2.0, 10.0], [4.0, 30.0], [6.0, 20.0]])
    assert np.asarray(r["scaled"]) == pytest.approx(np.asarray([[0.0, 0.0], [0.5, 1.0], [1.0, 0.5]]))


def test_grmms_rejects_a_constant_column():
    with pytest.raises(ValueError):
        geron_minmax_scaler([[1.0, 5.0], [2.0, 5.0]])


def test_grimp_median_resists_an_outlier_the_mean_does_not():
    X = [[1.0], [2.0], [300.0], [float("nan")]]
    assert geron_simple_imputer(X, "mean")["statistics"] == pytest.approx([303 / 3])
    assert geron_simple_imputer(X, "median")["statistics"] == pytest.approx([2.0])


def test_grimp_rejects_an_all_missing_column():
    with pytest.raises(ValueError):
        geron_simple_imputer([[float("nan")], [float("nan")]])


def test_grmlb_micro_f1_from_pooled_counts():
    S = [[0.9, 0.1], [0.2, 0.8]]
    Y = [[1, 1], [0, 1]]
    r = geron_multilabel_classification(S, Y)
    tp, fp, fn = 2, 0, 1              # pooled over both labels
    assert r["micro_f1"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
    assert r["hamming_loss"] == pytest.approx(1 / 4)


def test_grevr_uses_squared_singular_values():
    r = geron_explained_variance_ratio([3.0, 4.0])
    assert r["explained_variance_ratio"] == pytest.approx([9 / 25, 16 / 25])
    assert r["cumulative"][-1] == pytest.approx(1.0)


def test_grjll_matches_the_closed_form():
    m, eps = 5000, 0.2
    hand = 4 * math.log(m) / (eps**2 / 2 - eps**3 / 3)
    assert geron_johnson_lindenstrauss_bound(m, eps)["min_dimension"] == math.ceil(hand)


def test_grjll_grows_like_eps_to_the_minus_two():
    a = geron_johnson_lindenstrauss_bound(10000, 0.2)["min_dimension"]
    b = geron_johnson_lindenstrauss_bound(10000, 0.1)["min_dimension"]
    assert 3.5 < b / a < 4.5


def test_grgrp_variance_within_three_sigma_of_one_over_d():
    X = [[1.0] * 20 for _ in range(5)]
    d = 50
    r = geron_gaussian_random_projection(X, d=d, seed=3)
    n = 20 * d
    target = 1.0 / d
    three_sigma = 3 * target * math.sqrt(2.0 / (n - 1))
    assert abs(r["achieved_variance"] - target) < three_sigma


def test_grgrp_projection_is_the_matrix_product():
    X = [[1.0, 2.0, 3.0]]
    r = geron_gaussian_random_projection(X, d=2, seed=11)
    R = np.asarray(r["R"])
    assert np.asarray(r["projected"]) == pytest.approx(np.asarray(X) @ R)


# --------------------------------------------------------------------------
# trees and ensembles
# --------------------------------------------------------------------------

def test_grgin_matches_the_probability_interpretation():
    y = [0] * 3 + [1] * 2
    p = [3 / 5, 2 / 5]
    assert geron_gini_impurity(y)["gini"] == pytest.approx(1 - sum(v * v for v in p))
    # equals the chance two independent draws differ
    assert geron_gini_impurity(y)["gini"] == pytest.approx(2 * p[0] * p[1])


def test_grent_fair_coin_is_one_bit_and_pure_node_is_zero():
    assert geron_shannon_entropy([0, 1])["entropy"] == pytest.approx(1.0)
    assert geron_shannon_entropy([5, 5, 5])["entropy"] == 0.0


def test_grent_matches_hand_log_sum():
    y = [0] * 3 + [1] * 1
    hand = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    assert geron_shannon_entropy(y)["entropy"] == pytest.approx(hand)


def test_grig_matches_weighted_child_entropies():
    y = [0, 0, 0, 1]
    mask = [True, True, False, False]
    parent = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    right = 1.0                      # [0, 1]
    hand = parent - 0.5 * 0.0 - 0.5 * right
    assert geron_information_gain(y, mask)["information_gain"] == pytest.approx(hand)


def test_grig_rejects_a_one_sided_split():
    with pytest.raises(ValueError):
        geron_information_gain([0, 1], [True, True])


def test_grfim_normalises_each_tree_before_averaging():
    r = geron_feature_importance_mdi([[10.0, 0.0], [0.0, 1.0]])
    assert r["importance"] == pytest.approx([0.5, 0.5])
    assert sum(r["importance"]) == pytest.approx(1.0)


def test_grgbm_residual_and_shrinkage_by_hand():
    X = [[1.0], [2.0], [3.0], [4.0]]
    y = [0.0, 0.0, 2.0, 2.0]
    r = geron_gradient_boosting_residual(X, y, [1.0, 1.0, 1.0, 1.0], learning_rate=0.5)
    assert r["residuals"] == pytest.approx([-1.0, -1.0, 1.0, 1.0])
    assert r["F_new"] == pytest.approx([0.5, 0.5, 1.5, 1.5])
    assert r["mse_after"] < r["mse_before"]


def test_grgbm_enforces_the_learner_contract():
    with pytest.raises(ValueError):
        geron_gradient_boosting_residual([[1.0], [2.0]], [0.0, 1.0], 0.0,
                                         learner=lambda X, r: [1.0])


# --------------------------------------------------------------------------
# clustering, mixtures, outliers
# --------------------------------------------------------------------------

def test_grkmo_matches_hand_squared_distances():
    X = [[0.0, 0.0], [3.0, 4.0]]
    r = geron_kmeans_objective(X, [[0.0, 0.0]], [0, 0])
    assert r["inertia"] == pytest.approx(0.0 + 25.0)
    assert r["distances"] == pytest.approx([0.0, 5.0])


def test_grkmo_rejects_out_of_range_labels():
    with pytest.raises(ValueError):
        geron_kmeans_objective([[0.0]], [[0.0]], [1])


def test_grkmpp_picks_distinct_far_apart_points():
    X = [[0.0], [0.5], [100.0], [100.5]]
    r = geron_kmeans_pp_seeding(X, k=2, seed=0)
    assert len(set(r["indices"])) == 2
    assert r["min_pairwise_distance"] > 50
    for c in r["centroids"]:
        assert c in X


def test_grlof_flags_the_isolated_point():
    X = [[0.0], [0.5], [1.0], [1.5], [40.0]]
    r = geron_local_outlier_factor(X, k=2)
    assert r["most_outlying"] == 4
    assert r["lof"][4] > 3 * max(r["lof"][:4])


def test_grgmll_matches_hand_mixture_density():
    X = [[0.0]]
    pi = [0.3, 0.7]
    mus = [[0.0], [1.0]]
    covs = [[[1.0]], [[4.0]]]

    def npdf(x, mu, var):
        return math.exp(-((x - mu) ** 2) / (2 * var)) / math.sqrt(2 * math.pi * var)

    hand = math.log(0.3 * npdf(0, 0, 1) + 0.7 * npdf(0, 1, 4))
    assert geron_gmm_log_likelihood(X, pi, mus, covs)["log_likelihood"] == pytest.approx(hand)


def test_grgmll_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(ValueError):
        geron_gmm_log_likelihood([[0.0]], [0.3], [[0.0]], [[[1.0]]])


def test_grgmem_responsibilities_sum_to_one_and_em_does_not_decrease():
    X = [[0.0], [0.4], [5.0], [5.5]]
    r = geron_gmm_em_step(X, [0.5, 0.5], [[1.0], [4.0]], [[[1.0]], [[1.0]]])
    for row in r["responsibilities"]:
        assert sum(row) == pytest.approx(1.0)
    assert r["log_likelihood_after"] >= r["log_likelihood_before"]
    assert sum(r["pi_new"]) == pytest.approx(1.0)


def test_grkpc_centres_the_gram_matrix_and_reconstructs_it():
    X = [[0.0], [1.0], [2.5], [4.0]]
    r = geron_kernel_pca_rbf(X, gamma=0.3, d=4)
    Kc = np.asarray(r["kernel_centered"])
    assert np.abs(Kc.sum(axis=0)).max() < 1e-10
    Z = np.asarray(r["projected"])
    assert np.abs(Z @ Z.T - Kc).max() < 1e-8


# --------------------------------------------------------------------------
# optimizers, regularisation, schedules
# --------------------------------------------------------------------------

def test_grmom_first_step_equals_plain_gradient_descent():
    r = geron_momentum_update([1.0, 2.0], [0.4, -0.2], [0.0, 0.0], eta=0.1, beta=0.9)
    assert r["theta_new"] == pytest.approx([1.0 - 0.1 * 0.4, 2.0 + 0.1 * 0.2])


def test_grmom_velocity_follows_the_geometric_series():
    v = np.zeros(1)
    theta = np.zeros(1)
    g = np.array([1.0])
    beta = 0.5
    for k in range(1, 6):
        r = geron_momentum_update(theta, g, v, eta=0.1, beta=beta)
        theta = np.asarray(r["theta_new"])
        v = np.asarray(r["v_new"])
        closed = sum(beta**i for i in range(k))       # sum_{i<k} beta^i * g
        assert v[0] == pytest.approx(closed)
    assert r["terminal_speedup"] == pytest.approx(1 / (1 - beta))


def test_grmom_rejects_beta_of_one():
    with pytest.raises(ValueError):
        geron_momentum_update([1.0], [1.0], [0.0], eta=0.1, beta=1.0)


def test_grgcl_caps_the_norm_and_keeps_the_direction():
    g = [3.0, -4.0, 12.0]
    c = 2.0
    r = geron_gradient_clipping(g, c=c)
    clipped = np.asarray(r["clipped"])
    assert np.linalg.norm(clipped) <= c + 1e-12
    assert np.linalg.norm(clipped) == pytest.approx(c)
    cos = float(np.dot(g, clipped) / (np.linalg.norm(g) * np.linalg.norm(clipped)))
    assert cos == pytest.approx(1.0)


def test_grgcl_leaves_a_short_gradient_alone():
    r = geron_gradient_clipping([0.1, 0.1], c=1.0)
    assert r["clipped"] == pytest.approx([0.1, 0.1])
    assert r["was_clipped"] is False


def test_grgcl_treats_a_list_of_tensors_as_one_global_vector():
    r = geron_gradient_clipping([[3.0, 0.0], [0.0, 4.0]], c=1.0)
    flat = np.concatenate([np.asarray(t).ravel() for t in r["clipped"]])
    assert np.linalg.norm(flat) == pytest.approx(1.0)
    assert r["total_norm"] == pytest.approx(5.0)


def test_grmnr_projects_only_the_rows_that_exceed_r():
    r = geron_max_norm_regularization([[6.0, 8.0], [0.3, 0.4]], r=2.0)
    assert r["norms_after"] == pytest.approx([2.0, 0.5])
    assert r["rows_projected"] == [0]
    assert r["W_new"][0] == pytest.approx([1.2, 1.6])


def test_grdro_preserves_the_expectation_and_scale():
    a = [1.0] * 4000
    r = geron_dropout(a, p=0.25, seed=7)
    out = np.asarray(r["output"])
    assert r["scale"] == pytest.approx(1 / 0.75)
    kept = out[out != 0.0]
    assert kept == pytest.approx(np.full(kept.size, 4 / 3))
    assert abs(out.mean() - 1.0) < 0.05          # inverted dropout keeps E[a]


def test_grdro_at_p_zero_is_the_identity():
    r = geron_dropout([0.3, -2.0], p=0.0)
    assert r["output"] == pytest.approx([0.3, -2.0])


def test_grhei_variance_within_three_sigma_of_two_over_fan_in():
    fan_in = 100
    r = geron_he_init(fan_in, 100, seed=0)
    n = fan_in * 100
    target = 2.0 / fan_in
    three_sigma = 3 * target * math.sqrt(2.0 / (n - 1))
    assert r["target_variance"] == pytest.approx(target)
    assert abs(r["achieved_variance"] - target) < three_sigma
    assert abs(r["achieved_mean"]) < 3 * math.sqrt(target / n)


def test_grhei_doubling_fan_in_halves_the_variance():
    assert (geron_he_init(200)["target_variance"]
            == pytest.approx(geron_he_init(100)["target_variance"] / 2))


def test_grln_output_has_zero_mean_and_unit_variance():
    x = [3.0, -1.0, 7.0, 0.0]
    r = geron_layer_normalization(x, eps=0.0)
    z = np.asarray(r["normalized"])
    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert z.var() == pytest.approx(1.0)
    mu, sd = np.mean(x), np.std(x)
    assert z == pytest.approx([(v - mu) / sd for v in x])


def test_grln_normalises_rows_independently():
    r = geron_layer_normalization([[1.0, 3.0], [100.0, 300.0]], eps=0.0)
    assert r["normalized"][0] == pytest.approx(r["normalized"][1])


def test_grlrco_matches_the_closed_form_and_is_monotone():
    curve = geron_lr_cosine_annealing(0.01, 0.1, t=3, T=10)
    hand = [0.01 + 0.5 * 0.09 * (1 + math.cos(math.pi * t / 10)) for t in range(11)]
    assert curve["schedule"] == pytest.approx(hand)
    assert curve["eta"] == pytest.approx(hand[3])
    assert curve["is_monotone_decreasing"] is True
    assert all(b <= a + 1e-15 for a, b in zip(hand, hand[1:]))


def test_grlrco_endpoints_are_exact():
    c = geron_lr_cosine_annealing(0.0, 1.0, t=0, T=8)["schedule"]
    assert c[0] == pytest.approx(1.0)
    assert c[-1] == pytest.approx(0.0, abs=1e-15)


def test_grlrex_matches_the_closed_form_and_half_life():
    r = geron_lr_exponential_schedule(0.2, 0.8, t=6)
    assert r["schedule"] == pytest.approx([0.2 * 0.8**t for t in range(7)])
    assert r["is_monotone_decreasing"] is True
    hl = r["half_life"]
    assert 0.2 * 0.8**hl == pytest.approx(0.1)


def test_grlrex_rejects_a_growing_gamma():
    with pytest.raises(ValueError):
        geron_lr_exponential_schedule(0.1, 1.5, t=3)


# --------------------------------------------------------------------------
# training loops, splits, search
# --------------------------------------------------------------------------

def test_grdlm_epoch_is_a_permutation():
    r = geron_dataloader_minibatch(10, 3, shuffle=True, seed=4)
    flat = [i for b in r["batches"] for i in b]
    assert sorted(flat) == list(range(10))
    assert r["batch_sizes"] == [3, 3, 3, 1]
    assert r["covers_all"] is True


def test_grdlm_drop_last_discards_only_the_short_batch():
    r = geron_dataloader_minibatch(10, 3, shuffle=False, drop_last=True)
    assert [len(b) for b in r["batches"]] == [3, 3, 3]


def test_grmgd_single_full_batch_step_matches_hand_gradient():
    r = geron_minibatch_gradient_descent(X_LINE, Y_LINE, [0.0, 0.0], eta=0.01, b=4,
                                         n_iter=1, seed=0)
    X = np.asarray(X_LINE)
    g = (2 / 4) * X.T @ (X @ np.zeros(2) - np.asarray(Y_LINE))
    assert r["theta"] == pytest.approx((-0.01 * g).tolist())


def test_grmgd_converges_toward_the_normal_equation_solution():
    r = geron_minibatch_gradient_descent(X_LINE, Y_LINE, [0.0, 0.0], eta=0.03, b=2,
                                         n_iter=400, seed=1)
    exact = geron_ch4_normal_equation(X_LINE, Y_LINE)["theta"]
    assert r["theta"] == pytest.approx(exact, abs=0.05)
    assert r["final_cost"] < r["initial_cost"]


def test_greast_returns_the_argmin_snapshot():
    Xtr = [[1.0, float(i)] for i in range(6)]
    ytr = [float(i) + 4.0 for i in range(6)]
    Xva = [[1.0, 9.0], [1.0, 10.0]]
    yva = [9.0, 10.0]
    r = geron_early_stopping(Xtr, ytr, Xva, yva, n_iter=300, eta=0.02)
    hist = r["val_rmse_history"]
    assert r["best_val_rmse"] == pytest.approx(min(hist))
    assert r["best_iteration"] == hist.index(min(hist))
    # the returned theta really does score that RMSE
    scored = geron_linreg_mse_cost(Xva, yva, r["theta"])["rmse"]
    assert scored == pytest.approx(r["best_val_rmse"])


def test_greast_needs_a_validation_set():
    with pytest.raises(ValueError):
        geron_early_stopping([[1.0]], [1.0], [], [], n_iter=1, eta=0.1)


def test_grkfd_validates_every_index_exactly_once():
    r = geron_kfold_cv(11, 4, shuffle=True, seed=2)
    seen = [i for _, va in r["splits"] for i in va]
    assert sorted(seen) == list(range(11))
    assert r["fold_sizes"] == [3, 3, 3, 2]
    for tr, va in r["splits"]:
        assert set(tr).isdisjoint(va)
        assert len(tr) + len(va) == 11


def test_grkfd_rejects_k_of_one():
    with pytest.raises(ValueError):
        geron_kfold_cv(10, 1)


def test_grgs_finds_the_argmax_of_a_known_scorer():
    def scorer(Xtr, ytr, Xva, yva, params):
        return -(params["a"] - 3) ** 2 - params["b"]

    r = geron_grid_search_cv([[0.0]] * 6, [0.0] * 6,
                             {"a": [1, 3, 5], "b": [0, 1]}, K=3, fit_score=scorer)
    assert r["best_params"] == {"a": 3, "b": 0}
    assert r["best_score"] == pytest.approx(0.0)
    assert r["n_fits"] == 6 * 3
    assert len(r["candidates"]) == 6


def test_grgs_rejects_a_scorer_that_returns_nonsense():
    with pytest.raises(ValueError):
        geron_grid_search_cv([[0.0]] * 4, [0.0] * 4, {"a": [1]}, K=2,
                             fit_score=lambda *a: float("nan"))


def test_grlrnc_is_exact_on_noise_free_linear_data():
    X = [[1.0, float(i)] for i in range(12)]
    y = [2.0 * i + 1 for i in range(12)]
    r = geron_learning_curves(X, y, n_splits=4)
    assert max(r["train_rmse"]) == pytest.approx(0.0, abs=1e-9)
    assert max(r["val_rmse"]) == pytest.approx(0.0, abs=1e-9)
    assert r["final_gap"] == pytest.approx(0.0, abs=1e-9)


def test_grlrnc_shows_a_gap_when_the_model_is_wrong():
    X = [[1.0, float(i)] for i in range(12)]
    y = [float(i * i) for i in range(12)]
    r = geron_learning_curves(X, y, n_splits=4)
    assert r["final_gap"] > 0
    assert r["train_rmse"][-1] > 0


# --------------------------------------------------------------------------
# CNN / RNN blocks
# --------------------------------------------------------------------------

def test_grmpl_matches_hand_windows():
    X = [[1.0, 5.0, 2.0], [3.0, 4.0, 0.0], [9.0, 1.0, 1.0]]
    r = geron_max_pooling(X, k=2, stride=1)
    hand = [[max(X[i][j], X[i][j + 1], X[i + 1][j], X[i + 1][j + 1]) for j in range(2)]
            for i in range(2)]
    assert r["output"] == hand
    assert r["output_shape"] == (2, 2)


def test_grmpl_never_pads_with_zero():
    r = geron_max_pooling([[-5.0, -2.0], [-9.0, -7.0]], k=2)
    assert r["output"] == [[-2.0]]


def test_grfcn_output_size_and_overlap_counts():
    r = geron_fcn_upsample([[1.0, 1.0]], [[1.0, 1.0, 1.0]], stride=2)
    assert r["output_shape"] == (1, (2 - 1) * 2 + 3)
    assert r["output"][0] == pytest.approx([1.0, 1.0, 2.0, 1.0, 1.0])
    assert r["uniform_coverage"] is False        # the middle cell got two copies


def test_grfmp_dim_and_bytes():
    r = geron_feature_map_dim(150, 100, 200, bytes_per_value=4, batch_size=100)
    assert r["dim"] == 150 * 100 * 200
    assert r["batch_bytes"] == 150 * 100 * 200 * 4 * 100


def test_gremb_lookup_equals_one_hot_times_table():
    E = np.asarray([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    ids = [2, 0]
    onehot = np.zeros((2, 3))
    onehot[0, 2] = onehot[1, 0] = 1.0
    got = np.asarray(geron_embedding_lookup(ids, E)["embeddings"])
    assert got == pytest.approx(onehot @ E)


def test_gremb_rejects_an_out_of_range_id():
    with pytest.raises(ValueError):
        geron_embedding_lookup([3], [[1.0], [2.0]])


def test_grlstc_gates_match_hand_sigmoids():
    W = [[0.0, 1.0]]           # acts on [h, x], picks up x only
    r = geron_lstm_cell([2.0], [0.0], [0.5], W, W, W, W, 0.0, 0.0, 0.0, 0.0)
    s = 1 / (1 + math.exp(-2.0))
    g = math.tanh(2.0)
    c = s * 0.5 + s * g
    assert r["f"] == pytest.approx([s])
    assert r["c"] == pytest.approx([c])
    assert r["h"] == pytest.approx([s * math.tanh(c)])


def test_grlstc_closed_forget_gate_wipes_memory():
    W = [[0.0, 0.0]]
    r = geron_lstm_cell([0.0], [0.0], [3.0], W, W, W, W, -60.0, -60.0, 0.0, 0.0)
    assert r["c"] == pytest.approx([0.0], abs=1e-12)


def test_grgruc_matches_hand_gates():
    Wz = [[0.0, 1.0]]
    Wr = [[0.0, 2.0]]
    W = [[1.0, 0.0]]           # candidate reads r*h only
    r = geron_gru_cell([1.0], [0.6], Wz, Wr, W)
    z = 1 / (1 + math.exp(-1.0))
    rr = 1 / (1 + math.exp(-2.0))
    ht = math.tanh(rr * 0.6)
    assert r["z"] == pytest.approx([z])
    assert r["r"] == pytest.approx([rr])
    assert r["h"] == pytest.approx([(1 - z) * 0.6 + z * ht])


def test_grgruc_open_carry_when_z_is_zero():
    Z = [[0.0, 0.0]]
    r = geron_gru_cell([1.0], [0.9], [[0.0, -60.0]], Z, Z)
    assert r["h"] == pytest.approx([0.9])


# --------------------------------------------------------------------------
# attention
# --------------------------------------------------------------------------

def test_grmha_single_head_matches_hand_softmax_attention():
    I = [[1.0, 0.0], [0.0, 1.0]]
    K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[1.0, 0.0], [0.0, 1.0]]
    r = geron_multi_head_attention([[1.0, 0.0]], K, V, I, I, I, I, h=1)
    s = 1 / math.sqrt(2)
    w = [math.exp(s), math.exp(0.0)]
    tot = sum(w)
    assert r["output"][0] == pytest.approx([w[0] / tot, w[1] / tot])


def test_grmha_attention_rows_are_distributions():
    I = [[1.0, 0.0], [0.0, 1.0]]
    r = geron_multi_head_attention([[1.0, 2.0], [0.0, 1.0]],
                                   [[1.0, 0.0], [0.0, 1.0]],
                                   [[1.0, 0.0], [0.0, 1.0]], I, I, I, I, h=2)
    for A in r["attention_weights"]:
        for row in A:
            assert sum(row) == pytest.approx(1.0)
    assert r["d_head"] == 1


def test_grmha_rejects_head_count_that_does_not_divide_d_model():
    I = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError):
        geron_multi_head_attention([[1.0, 0.0]], I, I, I, I, I, I, h=3)


def test_grflash_equals_hand_computed_softmax_attention():
    Q = [[1.0, 0.5]]
    K = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    V = [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]
    scale = 1 / math.sqrt(2)
    logits = [scale * (Q[0][0] * k[0] + Q[0][1] * k[1]) for k in K]
    mx = max(logits)
    w = [math.exp(v - mx) for v in logits]
    tot = sum(w)
    hand = [sum(wi * v[j] for wi, v in zip(w, V)) / tot for j in range(2)]
    for bs in (1, 2, 3, 5):
        r = geron_flash_attention_tile(Q, K, V, block_size=bs)
        assert r["output"][0] == pytest.approx(hand)


def test_grflash_peak_memory_is_smaller_than_the_full_matrix():
    K = [[1.0, 0.0]] * 8
    r = geron_flash_attention_tile([[1.0, 0.0]], K, K, block_size=2)
    assert r["peak_score_elements"] < r["full_score_elements"]
    assert r["n_blocks"] == 4


def test_grflam_gate_is_tanh_and_zero_alpha_is_identity():
    I = [[1.0, 0.0], [0.0, 1.0]]
    h = [[0.3, -0.4]]
    vis = [[2.0, 6.0], [4.0, 2.0]]
    zero = geron_flamingo_cross_modal_attn(h, vis, 0.0, {"WQ": I, "WK": I, "WV": I})
    assert np.asarray(zero["h_new"]) == pytest.approx(np.asarray(h))
    open_ = geron_flamingo_cross_modal_attn([[0.0, 0.0]], vis, 0.8,
                                            {"WQ": I, "WK": I, "WV": I})
    # zero query -> uniform attention -> mean visual vector
    mean_vis = [3.0, 4.0]
    assert open_["gate"] == pytest.approx(math.tanh(0.8))
    assert open_["h_new"][0] == pytest.approx([math.tanh(0.8) * v for v in mean_vis])


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def test_gredsq_encodes_once_and_threads_the_context():
    calls = []

    def enc(x):
        calls.append("enc")
        return [float(sum(x))]

    def dec(y_prev, c, t):
        calls.append(f"dec{t}")
        return c[0] * t

    r = geron_encoder_decoder_seq2seq(enc, dec, [1.0, 2.0, 3.0], max_out_len=3)
    assert calls == ["enc", "dec1", "dec2", "dec3"]
    assert r["outputs"] == pytest.approx([6.0, 12.0, 18.0])


def test_gredsq_enforces_the_decoder_contract():
    with pytest.raises(ValueError):
        geron_encoder_decoder_seq2seq(lambda x: [1.0],
                                      lambda y, c, t: float("inf"),
                                      [1.0], max_out_len=2)


def test_grinc_prompt_order_and_shot_count():
    ex = [("a", "1"), ("b", "2")]
    r = geron_in_context_learning(ex, "c")
    assert r["prompt"] == "a -> 1\nb -> 2\nc ->"
    assert r["k_shot"] == 2
    assert r["example_order"] == ["a", "b"]
    swapped = geron_in_context_learning(list(reversed(ex)), "c")["prompt"]
    assert swapped != r["prompt"]


def test_grinc_rejects_a_model_that_returns_none():
    with pytest.raises(ValueError):
        geron_in_context_learning([("a", "1")], "b", predict=lambda p: None)


# --------------------------------------------------------------------------
# language-model losses
# --------------------------------------------------------------------------

def test_grgptl_matches_hand_log_sum_and_perplexity():
    logits = [[2.0, 0.0, -1.0], [0.0, 1.0, 1.0]]
    targets = [0, 2]
    l0 = 2.0 - math.log(math.exp(2.0) + 1 + math.exp(-1.0))
    l1 = 1.0 - math.log(1 + math.exp(1.0) + math.exp(1.0))
    r = geron_gpt_autoregressive_loss(logits, targets)
    assert r["loss"] == pytest.approx(-(l0 + l1))
    assert r["perplexity"] == pytest.approx(math.exp(-(l0 + l1) / 2))


def test_grgptl_uniform_perplexity_equals_vocab_size():
    r = geron_gpt_autoregressive_loss([[0.0] * 7], [3])
    assert r["perplexity"] == pytest.approx(7.0)


def test_grmlm_scores_only_masked_positions():
    logits = [[0.0, 0.0], [20.0, 0.0], [0.0, 0.0]]
    full = geron_bert_mlm_loss(logits, [0, 1, 1], [True, True, True])["loss"]
    part = geron_bert_mlm_loss(logits, [0, 1, 1], [True, False, True])["loss"]
    assert part == pytest.approx(2 * math.log(2))
    assert full > part


def test_grmlm_requires_at_least_one_mask():
    with pytest.raises(ValueError):
        geron_bert_mlm_loss([[0.0, 0.0]], [0], [False])


def test_grkdl_matches_hand_ce_and_kl():
    s = [1.0, 0.0]
    t = [2.0, 0.0]
    T, alpha = 2.0, 0.4
    ps = [math.exp(1.0), 1.0]
    ps = [v / sum(ps) for v in ps]
    ce = -math.log(ps[0])
    qs = [math.exp(0.5), 1.0]
    qs = [v / sum(qs) for v in qs]
    qt = [math.exp(1.0), 1.0]
    qt = [v / sum(qt) for v in qt]
    kl = sum(a * math.log(a / b) for a, b in zip(qs, qt))
    r = geron_knowledge_distillation_loss(s, t, 0, alpha=alpha, T=T)
    assert r["ce_hard"] == pytest.approx(ce)
    assert r["kl_soft"] == pytest.approx(T**2 * kl)
    assert r["loss"] == pytest.approx((1 - alpha) * ce + alpha * T**2 * kl)


def test_grkdl_identical_logits_give_zero_soft_loss():
    r = geron_knowledge_distillation_loss([1.0, -2.0], [1.0, -2.0], 0, alpha=1.0, T=3.0)
    assert r["kl_soft"] == pytest.approx(0.0, abs=1e-12)


def test_grdino_matches_hand_cross_entropy_and_sharpening():
    s = [0.5, 0.0]
    t = [1.0, 0.0]
    ts, tt = 0.5, 0.25
    ps = [math.exp(1.0), 1.0]
    ps = [v / sum(ps) for v in ps]
    pt = [math.exp(4.0), 1.0]
    pt = [v / sum(pt) for v in pt]
    hand = -sum(a * math.log(b) for a, b in zip(pt, ps))
    r = geron_dino_self_distillation(s, t, ts, tt)
    assert r["loss"] == pytest.approx(hand)
    assert r["teacher_entropy"] < r["student_entropy"]      # teacher is sharper


def test_grdino_centering_is_shift_invariant():
    a = geron_dino_self_distillation([0.0, 0.0], [1.0, 0.0], 0.1, 0.05)
    b = geron_dino_self_distillation([0.0, 0.0], [1.0, 0.0], 0.1, 0.05, center=[7.0, 7.0])
    assert a["loss"] == pytest.approx(b["loss"])


def test_grdpo_matches_hand_logsigmoid():
    r = geron_dpo_loss([-0.5], [-2.0], [-1.0], [-1.5], beta=0.5)
    margin = (-0.5 + 1.0) - (-2.0 + 1.5)
    assert r["margin"] == pytest.approx([margin])
    assert r["loss"] == pytest.approx(-math.log(1 / (1 + math.exp(-0.5 * margin))))


def test_grdpo_rejects_positive_log_probabilities():
    with pytest.raises(ValueError):
        geron_dpo_loss([0.5], [-1.0], [0.0], [0.0])


def test_grkldg_matches_hand_closed_form():
    mu = [0.5, -1.0]
    lv = [0.2, -0.3]
    hand = -0.5 * sum(1 + l - m**2 - math.exp(l) for m, l in zip(mu, lv))
    r = geron_kl_divergence_gaussian(mu, lv)
    assert r["kl"] == pytest.approx(hand)
    assert r["kl"] >= 0


def test_grkldg_is_zero_only_at_the_prior():
    assert geron_kl_divergence_gaussian([0.0, 0.0], [0.0, 0.0])["kl"] == 0.0
    assert geron_kl_divergence_gaussian([0.0], [0.5])["kl"] > 0


# --------------------------------------------------------------------------
# diffusion and GANs
# --------------------------------------------------------------------------

def test_grdpmf_matches_hand_coefficients():
    r = geron_ddpm_forward_process([2.0, -4.0], 1, [1.0, 0.36], noise=[1.0, 0.5])
    assert r["x_t"] == pytest.approx([0.6 * 2 + 0.8 * 1.0, 0.6 * -4 + 0.8 * 0.5])
    assert r["signal_coef"] ** 2 + r["noise_coef"] ** 2 == pytest.approx(1.0)


def test_grdpmf_rejects_an_increasing_alpha_bar():
    with pytest.raises(ValueError):
        geron_ddpm_forward_process([1.0], 1, [0.5, 0.9])


def test_grdpml_is_the_mean_squared_error_of_the_noise():
    eps = [0.3, -1.0, 0.5]
    pred = [0.0, -0.5, 0.5]
    hand = sum((a - b) ** 2 for a, b in zip(eps, pred)) / 3
    assert geron_ddpm_simple_loss(eps, pred)["loss"] == pytest.approx(hand)


def test_grdpmr_matches_hand_reverse_step():
    a, ab = 0.8, 0.64
    r = geron_ddpm_reverse_step([1.5], 0, [0.25], [a], [ab], sigma=0.0)
    coef = (1 - a) / math.sqrt(1 - ab)
    assert r["x_prev"] == pytest.approx([(1.5 - coef * 0.25) / math.sqrt(a)])


def test_grdpmr_sigma_zero_is_deterministic():
    a = geron_ddpm_reverse_step([1.0], 0, [0.1], [0.9], [0.5], sigma=0.0)
    b = geron_ddpm_reverse_step([1.0], 0, [0.1], [0.9], [0.5], sigma=0.0, seed=999)
    assert a["x_prev"] == pytest.approx(b["x_prev"])


def test_grgan_value_at_equilibrium_is_minus_two_log_two():
    r = geron_gan_minimax([1.0, 2.0], [3.0, 4.0], [0.5, 0.5], [0.5, 0.5])
    assert r["value"] == pytest.approx(-2 * math.log(2))
    assert r["at_equilibrium"] is True


def test_grgan_nonsaturating_loss_is_larger_when_the_generator_is_losing():
    losing = geron_gan_minimax([1.0], [0.0], [0.99], [0.01])
    even = geron_gan_minimax([1.0], [0.0], [0.5], [0.5])
    assert losing["g_loss_nonsaturating"] == pytest.approx(-math.log(0.01))
    assert losing["g_loss_nonsaturating"] > even["g_loss_nonsaturating"]


def test_grmcol_counts_modes_and_off_distribution_samples():
    modes = [[0.0], [10.0], [20.0]]
    samples = [[0.05], [-0.05], [10.1], [5.0]]
    r = geron_gan_mode_collapse_metric(samples, modes, tol=1.0)
    assert r["samples_per_mode"] == [2, 1, 0]
    assert r["coverage"] == pytest.approx(2 / 3)
    assert r["mode_collapse_rate"] == pytest.approx(1 / 3)
    assert r["n_off_distribution"] == 1


# --------------------------------------------------------------------------
# reinforcement learning
# --------------------------------------------------------------------------

def test_grepl_probabilities_sum_to_one_and_greedy_gets_the_extra_mass():
    Q = [1.0, 7.0, 3.0, 2.0]
    eps = 0.4
    r = geron_epsilon_greedy(Q, eps)
    assert sum(r["action_probabilities"]) == pytest.approx(1.0)
    assert r["greedy_probability"] == pytest.approx(1 - eps + eps / 4)
    assert r["action_probabilities"][0] == pytest.approx(eps / 4)


def test_grepl_frequency_matches_the_stated_distribution():
    Q = [0.0, 1.0]
    eps = 0.5
    # consecutive seeds move the LCG's first output by only 1664525/2**32,
    # so the seeds are spread out to sweep the whole unit interval
    hits = sum(geron_epsilon_greedy(Q, eps, seed=s * 7919)["action"] for s in range(600))
    assert abs(hits / 600 - (1 - eps + eps / 2)) < 0.06


def test_grdqnl_matches_hand_targets():
    Q = [[0.5, 0.0], [0.0, 0.0]]
    QT = [[0.0, 0.0], [4.0, 1.0]]
    r = geron_dqn_loss(Q, QT, [(0, 0, 1.0, 1, False)], gamma=0.5)
    target = 1.0 + 0.5 * 4.0
    assert r["targets"] == pytest.approx([target])
    assert r["loss"] == pytest.approx((target - 0.5) ** 2)


def test_grdqnl_terminal_transition_drops_the_bootstrap():
    QT = [[0.0, 0.0], [100.0, 0.0]]
    r = geron_dqn_loss([[0.0, 0.0], [0.0, 0.0]], QT,
                       [(0, 0, 2.0, 1, True)], gamma=0.99)
    assert r["targets"] == pytest.approx([2.0])


def test_grduel_advantages_centre_and_q_mean_is_v():
    r = geron_dueling_dqn([2.0, -1.0], [[1.0, 4.0, 1.0], [0.0, 0.0, 3.0]])
    for row in r["centered_advantage"]:
        assert sum(row) == pytest.approx(0.0)
    for v, q in zip([2.0, -1.0], r["Q"]):
        assert sum(q) / 3 == pytest.approx(v)


def test_grduel_is_invariant_to_shifting_the_advantage_stream():
    a = geron_dueling_dqn([1.0], [[0.0, 5.0]])["Q"]
    b = geron_dueling_dqn([1.0], [[100.0, 105.0]])["Q"]
    assert np.asarray(a) == pytest.approx(np.asarray(b))


# --------------------------------------------------------------------------
# efficiency
# --------------------------------------------------------------------------

def test_grdyq_round_trips_within_the_quantization_grid():
    x = [[0.5, -1.0], [0.25, 1.0]]
    w = [[2.0], [-2.0]]
    r = geron_dynamic_quantization(x, w)
    assert r["scale_x"] == pytest.approx(1.0 / 127)
    assert r["scale_w"] == pytest.approx(2.0 / 127)
    ref = (np.asarray(x) @ np.asarray(w))
    assert np.abs(np.asarray(r["output"]) - ref).max() < 2 * r["scale_x"] * r["scale_w"] * 127


def test_grdyq_outlier_coarsens_the_grid():
    tight = geron_dynamic_quantization([[1.0, 1.0]], [[1.0], [1.0]])
    loose = geron_dynamic_quantization([[1.0, 500.0]], [[1.0], [0.0]])
    assert loose["scale_x"] > 100 * tight["scale_x"]
    assert loose["max_abs_error"] > tight["max_abs_error"]


def test_grfp6_scaling_round_trips_and_flags_overflow():
    r = geron_fp16_mixed_precision(0.5, 512.0, gradients=[1e-8, 0.25])
    assert r["loss_scaled"] == pytest.approx(256.0)
    assert r["scaled_gradients"] == pytest.approx([1e-8 * 512, 0.25 * 512])
    assert r["recovered_gradients"] == pytest.approx([1e-8, 0.25])
    assert r["n_underflow_before"] == 1
    assert geron_fp16_mixed_precision(2.0, 65536.0)["overflow"] is True


def test_grkvc_matches_the_closed_form_and_compression_ratio():
    args = dict(seq_len=512, num_layers=24, num_heads=16, d_head=64)
    hand = 512 * 24 * 16 * 64 * 2 * 16 // 8
    r16 = geron_kv_cache_compression(bits=16, **args)
    r4 = geron_kv_cache_compression(bits=4, **args)
    assert r16["cache_bytes"] == hand
    assert r4["cache_bytes"] == hand // 4
    assert r4["compression_ratio"] == pytest.approx(4.0)


def test_grkvc_is_linear_in_sequence_length():
    a = geron_kv_cache_compression(100, 4, 4, 8)["cache_bytes"]
    b = geron_kv_cache_compression(300, 4, 4, 8)["cache_bytes"]
    assert b == 3 * a


# --------------------------------------------------------------------------
# forward-mode autodiff
# --------------------------------------------------------------------------

def test_grfad_matches_analytic_derivatives():
    cases = [
        (lambda z: z * z * z, 2.0, 8.0, 12.0),
        (lambda z: z * z + 3.0 * z + 1.0, 2.0, 11.0, 7.0),
        (lambda z: (z * z).exp(), 1.0, math.e, 2 * math.e),
        (lambda z: 1.0 / z, 4.0, 0.25, -1 / 16),
        (lambda z: z.log(), 5.0, math.log(5.0), 0.2),
    ]
    for f, x, val, der in cases:
        r = geron_forward_mode_autodiff(x, 1.0, f)
        assert r["value"] == pytest.approx(val)
        assert r["derivative"] == pytest.approx(der)


def test_grfad_beats_the_finite_difference_it_reports():
    r = geron_forward_mode_autodiff(1.5, 1.0, lambda z: z**4)
    assert r["derivative"] == pytest.approx(4 * 1.5**3)      # exact
    assert r["check_abs_error"] < 1e-4                       # fd is only close


def test_grfad_seed_scales_the_derivative():
    a = geron_forward_mode_autodiff(3.0, 1.0, lambda z: z * z)["derivative"]
    b = geron_forward_mode_autodiff(3.0, 2.5, lambda z: z * z)["derivative"]
    assert b == pytest.approx(2.5 * a)


def test_grfad_rejects_a_function_that_does_not_return_a_dual():
    with pytest.raises(ValueError):
        geron_forward_mode_autodiff(1.0, 1.0, lambda z: 3.0)


def test_dual_product_rule_directly():
    d = Dual(2.0, 1.0) * Dual(5.0, 3.0)
    assert (d.value, d.deriv) == (10.0, 1.0 * 5.0 + 2.0 * 3.0)


# --------------------------------------------------------------------------
# anti-stub sweep: a mean-of-inputs body would pass none of these
# --------------------------------------------------------------------------

def test_no_module_returns_the_mean_of_its_inputs():
    """Each case: (label, estimate, the mean a stub body would return)."""
    xs = [1.0, 2.0, 3.0, 10.0]
    cases = [
        ("grmse", geron_linreg_mse_cost(X_LINE, Y_LINE, [0.0, 1.0])["estimate"],
         float(np.mean(Y_LINE))),
        ("grgin", geron_gini_impurity([0, 0, 1])["estimate"], float(np.mean([0, 0, 1]))),
        ("grent", geron_shannon_entropy([0, 0, 0, 1])["estimate"],
         float(np.mean([0, 0, 0, 1]))),
        ("grmae", geron_mae([0.0, 0.0], [1.0, 5.0])["estimate"], float(np.mean([0.0, 0.0]))),
        ("grevr", geron_explained_variance_ratio(xs)["estimate"][0], float(np.mean(xs))),
        ("grmom", geron_momentum_update([1.0, 2.0], [0.5, 0.5], [0.0, 0.0],
                                        eta=0.1)["estimate"], float(np.mean([1.0, 2.0]))),
        ("grhev", geron_heaviside_step(xs)["estimate"][0], float(np.mean(xs))),
        ("grkldg", geron_kl_divergence_gaussian([1.0, 2.0], [0.0, 0.0])["estimate"],
         float(np.mean([1.0, 2.0]))),
        ("grjll", geron_johnson_lindenstrauss_bound(1000, 0.1)["estimate"], 1000.0),
        ("grkvc", geron_kv_cache_compression(128, 4, 4, 16)["estimate"], 128.0),
        ("grdpml", geron_ddpm_simple_loss([1.0, 3.0], [0.0, 0.0])["estimate"],
         float(np.mean([1.0, 3.0]))),
        ("grgptl", geron_gpt_autoregressive_loss([[0.0, 0.0]], [0])["estimate"], 0.0),
        ("grf1", geron_f1_score([1, 1, 1, 0], [1, 0, 0, 0])["estimate"],
         float(np.mean([1, 1, 1, 0]))),
        ("grkmo", geron_kmeans_objective([[0.0], [4.0]], [[2.0]], [0, 0])["estimate"],
         float(np.mean([0.0, 4.0]))),
        ("grlrex", geron_lr_exponential_schedule(0.1, 0.5, t=3)["estimate"], 0.1),
    ]
    for label, est, stub in cases:
        assert est != pytest.approx(stub), f"{label} looks like a mean-of-inputs stub"
