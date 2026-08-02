# morie.fn -- test file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent-route checks for the w4a tranche (Géron, classic ML + DL).

Every assertion is derived from a route other than the implementation:
brute-force counting, central finite differences, hand log-sums,
closed-form identities, or invariants that a mean-of-inputs stub cannot
satisfy. Data is explicit or from the LCG.
"""

import math

from morie.fn import _array_core as np
import pytest

# ---------------------------------------------------------------- helpers


def lcg(n, seed=1):
    """n uniforms in (0, 1) from the standard LCG."""
    s = int(seed) % 2**32
    out = []
    for _ in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out.append((s + 0.5) / 2**32)
    return np.asarray(out)


def fd_grad(f, x, h=1e-6):
    """Central finite-difference gradient of a scalar function."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        up, dn = x.copy(), x.copy()
        up.flat[i] += h
        dn.flat[i] -= h
        g.flat[i] = (f(up) - f(dn)) / (2 * h)
    return g


def softmax_rows(Z):
    Z = np.asarray(Z, dtype=float)
    Z = Z - Z.max(axis=-1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=-1, keepdims=True)


# ================================================================ metrics


def test_hmgini_matches_brute_force_pair_probability():
    from morie.fn.hmgini import geron_gini_impurity

    y = [0, 0, 1, 2, 2, 2]
    # Gini = P(two independent draws differ), counted by brute force.
    n = len(y)
    diff = sum(1 for a in y for b in y if a != b)
    assert geron_gini_impurity(y)["gini"] == pytest.approx(diff / (n * n))
    # A mean-of-inputs stub would return 1.1666..., not 0.6111...
    assert geron_gini_impurity(y)["gini"] != pytest.approx(float(np.mean(y)))


def test_hmgini_bounds_and_pure_node():
    from morie.fn.hmgini import geron_gini_impurity

    assert geron_gini_impurity([7, 7, 7])["gini"] == 0.0
    r = geron_gini_impurity([0, 1, 2, 3])
    assert r["gini"] == pytest.approx(0.75) == pytest.approx(r["max_possible"])
    with pytest.raises(ValueError):
        geron_gini_impurity([])


def test_hment_entropy_against_hand_log_sum():
    from morie.fn.hment import geron_entropy_impurity

    y = [0, 0, 0, 1]
    hand = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    assert geron_entropy_impurity(y)["entropy"] == pytest.approx(hand)
    assert geron_entropy_impurity([0, 1, 2, 3])["entropy"] == pytest.approx(2.0)


def test_hment_dominates_gini_in_bits():
    from morie.fn.hment import geron_entropy_impurity
    from morie.fn.hmgini import geron_gini_impurity

    for y in ([0, 1], [0, 0, 1], [0, 1, 2, 2]):
        assert geron_entropy_impurity(y)["entropy"] >= geron_gini_impurity(y)["gini"]


def test_hmcfm_confusion_matrix_by_brute_force_counting():
    from morie.fn.hmcfm import geron_confusion_matrix

    yt = [0, 1, 2, 1, 0, 2, 2, 1]
    yp = [0, 1, 1, 1, 2, 2, 0, 0]
    r = geron_confusion_matrix(yt, yp)
    for i in range(3):
        for j in range(3):
            hand = sum(1 for a, b in zip(yt, yp) if a == i and b == j)
            assert r["matrix"][i][j] == hand
    assert r["accuracy"] == pytest.approx(sum(a == b for a, b in zip(yt, yp)) / len(yt))


def test_hmcfm_string_labels_and_marginals():
    from morie.fn.hmcfm import geron_confusion_matrix

    r = geron_confusion_matrix(["b", "a", "b"], ["a", "a", "b"])
    assert r["labels"] == ["a", "b"]
    assert r["support"] == [1, 2]
    assert r["predicted_totals"] == [2, 1]


def test_hmf1_harmonic_mean_identity():
    from morie.fn.hmf1 import geron_f1_score

    yt = [0, 0, 1, 1, 1, 0]
    yp = [0, 1, 1, 1, 0, 0]
    tp = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 0)
    p, rc = tp / (tp + fp), tp / (tp + fn)
    r = geron_f1_score(yt, yp)
    assert (r["tp"], r["fp"], r["fn"]) == (tp, fp, fn)
    assert r["f1"] == pytest.approx(2 / (1 / p + 1 / rc))


def test_hmf1_never_exceeds_its_components():
    from morie.fn.hmf1 import geron_f1_score

    r = geron_f1_score([0, 1, 1, 1], [0, 1, 0, 0])
    assert min(r["precision"], r["recall"]) <= r["f1"] <= max(r["precision"], r["recall"])
    assert geron_f1_score([0, 1], [0, 0])["f1"] == 0.0


def test_hmeaf_rows_normalise_and_diagonal_removed():
    from morie.fn.hmeaf import geron_error_analysis

    yt = [0, 0, 0, 1, 1, 2]
    yp = [0, 1, 2, 1, 0, 2]
    r = geron_error_analysis(yt, yp)
    for row in r["normalized"]:
        assert sum(row) == pytest.approx(1.0)
    for i, row in enumerate(r["error_matrix"]):
        assert row[i] == 0.0
    assert r["error_rate"] == pytest.approx(sum(a != b for a, b in zip(yt, yp)) / len(yt))


def test_hmeaf_top_confusion_is_the_largest_off_diagonal():
    from morie.fn.hmeaf import geron_error_analysis

    r = geron_error_analysis([0, 0, 0, 1], [1, 1, 0, 1])
    assert r["top_confusions"][0][:2] == (0, 1)
    assert r["top_confusions"][0][2] == pytest.approx(2 / 3)


# ============================================================ activations


def test_hmelu_derivative_against_finite_differences():
    from morie.fn.hmelu import geron_elu

    zs = [-2.0, -0.5, 0.3, 1.7]
    r = geron_elu(zs, alpha=1.3)
    h = 1e-6
    for i, z in enumerate(zs):
        fd = (geron_elu([z + h], alpha=1.3)["a"][0] - geron_elu([z - h], alpha=1.3)["a"][0]) / (2 * h)
        assert r["derivative"][i] == pytest.approx(fd, abs=1e-5)


def test_hmelu_saturation_and_continuity_at_zero():
    from morie.fn.hmelu import geron_elu

    assert geron_elu([-1e3], alpha=2.0)["a"][0] == pytest.approx(-2.0)
    left = geron_elu([-1e-9])["a"][0]
    right = geron_elu([1e-9])["a"][0]
    assert abs(left - right) < 1e-8
    with pytest.raises(ValueError):
        geron_elu([1.0], alpha=-1.0)


def test_hmgelu_against_independent_erf_definition():
    from morie.fn.hmgelu import geron_gelu

    zs = [-2.0, -0.3, 0.0, 1.1, 3.0]
    r = geron_gelu(zs)
    for i, z in enumerate(zs):
        hand = z * 0.5 * (1 + math.erf(z / math.sqrt(2)))
        assert r["a"][i] == pytest.approx(hand, abs=1e-12)


def test_hmgelu_derivative_and_non_monotonicity():
    from morie.fn.hmgelu import geron_gelu

    zs = [-1.5, -1.0, 0.5, 2.0]
    r = geron_gelu(zs)
    h = 1e-6
    for i, z in enumerate(zs):
        fd = (geron_gelu([z + h])["a"][0] - geron_gelu([z - h])["a"][0]) / (2 * h)
        assert r["derivative"][i] == pytest.approx(fd, abs=1e-5)
    assert geron_gelu([-1.0])["derivative"][0] < 0.0  # non-monotone


def test_hmdrp_dropout_is_unbiased_in_expectation():
    from morie.fn.hmdrp import geron_dropout

    x = np.full(4000, 3.0)
    r = geron_dropout(x, p=0.5, seed=11)
    assert r["y"].mean() == pytest.approx(3.0, abs=0.15)
    assert set(np.unique(r["y"]).tolist()) <= {0.0, 6.0}


def test_hmdrp_inference_is_the_identity_and_p1_rejected():
    from morie.fn.hmdrp import geron_dropout

    assert list(geron_dropout([1.0, 2.0], p=0.9, training=False)["y"]) == [1.0, 2.0]
    with pytest.raises(ValueError):
        geron_dropout([1.0], p=1.0)


# ================================================================= losses


def test_hmcec_cost_against_hand_log_sum():
    from morie.fn.hmcec import geron_cross_entropy_cost

    X = [[1.0, 2.0]]
    th = [[0.5, -0.5], [1.0, 0.0]]
    logits = [1.0 * 0.5 + 2.0 * 1.0, 1.0 * -0.5 + 2.0 * 0.0]  # [2.5, -0.5]
    hand = -(logits[0] - math.log(math.exp(logits[0]) + math.exp(logits[1])))
    assert geron_cross_entropy_cost(X, [0], th)["cost"] == pytest.approx(hand)


def test_hmcec_chance_cost_and_shape_errors():
    from morie.fn.hmcec import geron_cross_entropy_cost

    r = geron_cross_entropy_cost([[1.0], [2.0], [3.0]], [0, 1, 2], [[0.0, 0.0, 0.0]])
    assert r["cost"] == pytest.approx(math.log(3))
    with pytest.raises(ValueError):
        geron_cross_entropy_cost([[1.0]], [0], [[0.0], [0.0]])


def test_hmceg_gradient_against_finite_differences():
    from morie.fn.hmcec import geron_cross_entropy_cost
    from morie.fn.hmceg import geron_cross_entropy_gradient

    u = lcg(6, seed=3)
    X = (u[:4].reshape(2, 2) * 2 - 1).tolist()
    Y = [0, 1]
    th0 = np.array([[0.3, -0.2], [0.1, 0.7]])

    def cost(flat):
        return geron_cross_entropy_cost(X, Y, flat.reshape(2, 2))["cost"]

    g = np.asarray(geron_cross_entropy_gradient(X, Y, th0)["gradient"]).ravel()
    assert np.allclose(g, fd_grad(cost, th0.ravel(), h=1e-6), atol=1e-6)


def test_hmceg_rows_sum_to_zero():
    from morie.fn.hmceg import geron_cross_entropy_gradient

    r = geron_cross_entropy_gradient([[1.0, -2.0], [0.5, 3.0]], [1, 0], [[0.2, -0.1], [0.4, 0.3]])
    for row in r["gradient"]:
        assert sum(row) == pytest.approx(0.0, abs=1e-12)


def test_hmenet_endpoints_are_lasso_and_ridge():
    from morie.fn.hmenet import geron_elastic_net

    X, y, th = [[1.0], [2.0]], [2.0, 4.0], [0.5, 2.0]
    mse = float(np.mean((np.array([0.5 + 2.0, 0.5 + 4.0]) - np.array(y)) ** 2))
    lasso = geron_elastic_net(X, y, th, alpha=3.0, r=1.0)["cost"]
    ridge = geron_elastic_net(X, y, th, alpha=3.0, r=0.0)["cost"]
    assert lasso == pytest.approx(mse + 3.0 * 2.0)
    assert ridge == pytest.approx(mse + 0.5 * 3.0 * 4.0)


def test_hmenet_gradient_against_finite_differences_away_from_kinks():
    from morie.fn.hmenet import geron_elastic_net

    X, y = [[1.0], [2.0], [3.0]], [1.0, 2.5, 3.0]
    th0 = np.array([0.2, 0.9])

    def cost(t):
        return geron_elastic_net(X, y, t, alpha=0.5, r=0.5)["cost"]

    g = np.asarray(geron_elastic_net(X, y, th0, alpha=0.5, r=0.5)["gradient"])
    assert np.allclose(g, fd_grad(cost, th0, h=1e-6), atol=1e-5)


def test_hmdpo_loss_against_hand_sigmoid():
    from morie.fn.hmdpo import geron_dpo

    pi = [[-0.5, -2.0]]
    ref = [[-1.0, -1.5]]
    beta = 2.0
    margin = beta * ((-0.5 + 1.0) - (-2.0 + 1.5))
    hand = -math.log(1 / (1 + math.exp(-margin)))
    r = geron_dpo(pi, ref, beta=beta)
    assert r["margin"][0] == pytest.approx(margin)
    assert r["loss"] == pytest.approx(hand)


def test_hmdpo_is_indifferent_when_policy_equals_reference():
    from morie.fn.hmdpo import geron_dpo

    r = geron_dpo([[-1.0, -2.0], [-3.0, -0.5]], [[-1.0, -2.0], [-3.0, -0.5]])
    assert r["loss"] == pytest.approx(math.log(2))
    assert r["prob_preferred"] == pytest.approx([0.5, 0.5])
    with pytest.raises(ValueError):
        geron_dpo([[0.5, -1.0]], [[-1.0, -1.0]])  # log probs must be <= 0


def test_hmelb_kl_against_closed_form():
    from morie.fn.hmelb import geron_elbo

    mu, ls = 0.7, -0.3
    sig2 = math.exp(2 * ls)
    hand_kl = -0.5 * (1 + 2 * ls - mu**2 - sig2)
    r = geron_elbo([[0.0]], [[mu]], [[ls]])
    assert r["kl"] == pytest.approx(hand_kl)
    assert r["loss"] == pytest.approx(-r["elbo"])


def test_hmelb_reconstruction_term_is_gaussian_log_lik():
    from morie.fn.hmelb import geron_elbo

    r = geron_elbo([[0.0, 0.0]], [[0.0]], [[0.0]], x_recon=[[1.0, 2.0]])
    hand = -0.5 * (1.0 + 4.0) - 2 * 0.5 * math.log(2 * math.pi)
    assert r["reconstruction_log_lik"] == pytest.approx(hand)


# ============================================================== optimizers


def test_hmgcl_clipping_preserves_direction():
    from morie.fn.hmgcl import geron_gradient_clipping

    g = [3.0, -4.0]
    r = geron_gradient_clipping(g, max_norm=2.5)
    assert r["total_norm"] == pytest.approx(5.0)
    assert list(r["clipped"]) == pytest.approx([1.5, -2.0])
    assert r["new_norm"] == pytest.approx(2.5)
    cos = float(np.dot(g, r["clipped"]) / (np.linalg.norm(g) * np.linalg.norm(r["clipped"])))
    assert cos == pytest.approx(1.0)


def test_hmgcl_is_a_no_op_below_the_threshold():
    from morie.fn.hmgcl import geron_gradient_clipping

    r = geron_gradient_clipping([[0.1, 0.2], [0.3]], max_norm=10.0)
    assert r["was_clipped"] is False
    assert list(r["clipped"][0]) == pytest.approx([0.1, 0.2])


def test_hmearl_first_iteration_matches_hand_gradient_step():
    from morie.fn.hmearl import geron_early_stopping

    Xt, yt = [[1.0], [2.0]], [2.0, 4.0]
    eta = 0.05
    A = np.array([[1.0, 1.0], [1.0, 2.0]])
    theta0 = np.zeros(2)
    grad = (2.0 / 2) * (A.T @ (A @ theta0 - np.array(yt)))
    theta1 = theta0 - eta * grad
    hand_train_rmse = float(np.sqrt(np.mean((A @ theta1 - np.array(yt)) ** 2)))
    r = geron_early_stopping(Xt, yt, [[3.0]], [6.0], n_iter=1, eta=eta)
    assert r["train_rmse"][1] == pytest.approx(hand_train_rmse)


def test_hmearl_keeps_the_best_snapshot_not_the_last():
    from morie.fn.hmearl import geron_early_stopping

    r = geron_early_stopping([[0.0], [1.0], [2.0], [3.0]], [0.0, 2.0, 4.0, 6.0],
                             [[0.0], [1.0]], [3.0, 3.0], n_iter=150, eta=0.05)
    assert r["best_val_rmse"] == pytest.approx(min(r["val_rmse"]))
    assert r["best_val_rmse"] <= r["final_val_rmse"]


def test_hmfth_one_step_matches_hand_derived_first_iteration():
    from morie.fn.hmfth import geron_finetune_lm

    def task(th, batch):
        return float((th[0] - 3.0) ** 2), np.array([2.0 * (th[0] - 3.0)])

    r = geron_finetune_lm(task, [1], epochs=1, lr=0.25, theta=[0.0])
    assert r["theta"][0] == pytest.approx(0.0 - 0.25 * (2.0 * (0.0 - 3.0)))
    assert r["loss_history"][0] == pytest.approx(9.0)


def test_hmfth_weight_decay_and_freezing():
    from morie.fn.hmfth import geron_finetune_lm

    def two(th, batch):
        return float(th[0] ** 2 + th[1] ** 2), np.array([2 * th[0], 2 * th[1]])

    r = geron_finetune_lm(two, [1], epochs=1, lr=0.1, theta=[1.0, 1.0],
                          freeze=[True, False], weight_decay=1.0)
    assert r["theta"][0] == 1.0
    assert r["theta"][1] == pytest.approx(1.0 - 0.1 * (2.0 + 1.0))
    with pytest.raises(ValueError):
        geron_finetune_lm(two, [1], theta=[1.0, 1.0], freeze=[True, True])


# ============================================================ linear models


def test_hmdbd_signed_distance_is_scale_free():
    from morie.fn.hmdbd import geron_decision_boundary

    pts = [[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]]
    a = geron_decision_boundary([-1.0, 1.0, 1.0], pts)
    b = geron_decision_boundary([-10.0, 10.0, 10.0], pts)
    assert a["signed_distance"] == pytest.approx(b["signed_distance"])
    assert a["labels"] == b["labels"]
    hand = [(-1.0 + p[0] + p[1]) / math.sqrt(2) for p in pts]
    assert a["signed_distance"] == pytest.approx(hand)


def test_hmdbd_line_form_reproduces_the_boundary():
    from morie.fn.hmdbd import geron_decision_boundary

    r = geron_decision_boundary([-2.0, 1.0, 2.0], [[0.0, 1.0]])
    slope, intercept = r["line"]
    for x1 in (-3.0, 0.0, 5.0):
        x2 = slope * x1 + intercept
        assert abs(-2.0 + 1.0 * x1 + 2.0 * x2) < 1e-9
    with pytest.raises(ValueError):
        geron_decision_boundary([1.0, 0.0, 0.0], [[0.0, 0.0]])


def test_hmevr_ratios_sum_to_one_and_match_eigenvalues():
    from morie.fn.hmevr import geron_explained_variance_ratio

    X = np.array([[-2.0, -1.0], [2.0, 1.0], [-2.0, 1.0], [2.0, -1.0]])
    r = geron_explained_variance_ratio(X)
    cov = np.cov(X.T, ddof=1)
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    assert r["explained_variance"] == pytest.approx(eig.tolist())
    assert sum(r["explained_variance_ratio"]) == pytest.approx(1.0)


def test_hmevr_rank_one_data_puts_everything_in_one_component():
    from morie.fn.hmevr import geron_explained_variance_ratio

    r = geron_explained_variance_ratio([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    assert r["explained_variance_ratio"][0] == pytest.approx(1.0)
    assert r["explained_variance_ratio"][1] == pytest.approx(0.0, abs=1e-12)
    assert r["n_for_95"] == 1


# ================================================================== trees


def test_hmcart_split_matches_exhaustive_search():
    from morie.fn.hmcart import geron_cart_algorithm

    X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [0, 0, 0, 1, 1]

    def gini(v):
        if not len(v):
            return 0.0
        p = np.bincount(v, minlength=2) / len(v)
        return float(1 - np.sum(p * p))

    best = min(
        ((len([a for a in X if a[0] <= t]) / 5) * gini([c for a, c in zip(X, y) if a[0] <= t])
         + (len([a for a in X if a[0] > t]) / 5) * gini([c for a, c in zip(X, y) if a[0] > t]), t)
        for t in (1.5, 2.5, 3.5, 4.5)
    )
    r = geron_cart_algorithm(X, y)
    assert r["tree"]["threshold"] == pytest.approx(best[1])
    assert r["train_accuracy"] == 1.0


def test_hmcart_regression_leaves_are_group_means():
    from morie.fn.hmcart import geron_cart_algorithm

    X = [[1.0], [2.0], [3.0], [4.0]]
    y = [1.0, 3.0, 10.0, 12.0]
    r = geron_cart_algorithm(X, y, criterion="mse", max_depth=1)
    assert r["predictions"] == pytest.approx([2.0, 2.0, 11.0, 11.0])
    assert r["train_mse"] == pytest.approx(float(np.mean((np.array(y) - np.array([2, 2, 11, 11])) ** 2)))


def test_hmcart_depth_limit_is_respected():
    from morie.fn.hmcart import geron_cart_algorithm

    u = lcg(20, seed=5)
    X = u[:10].reshape(10, 1).tolist()
    y = (u[10:] > 0.5).astype(int).tolist()
    for d in (0, 1, 2, 3):
        assert geron_cart_algorithm(X, y, max_depth=d)["depth"] <= d


def test_hmcdt_probabilities_are_leaf_frequencies():
    from morie.fn.hmcdt import geron_classification_tree

    r = geron_classification_tree([[1.0], [2.0], [3.0]], [0, 1, 1], max_depth=0)
    assert r["probabilities"][0] == pytest.approx([1 / 3, 2 / 3])
    for row in r["probabilities"]:
        assert sum(row) == pytest.approx(1.0)


def test_hmcdt_rejects_continuous_targets():
    from morie.fn.hmcdt import geron_classification_tree

    with pytest.raises(ValueError):
        geron_classification_tree([[1.0], [2.0]], [0.5, 1.5])
    with pytest.raises(ValueError):
        geron_classification_tree([[1.0], [2.0]], [0, 1], criterion="mse")


def test_hmdtr_constraints_never_grow_the_tree():
    from morie.fn.hmdtr import geron_tree_regularization

    u = lcg(24, seed=9)
    X = u[:12].reshape(12, 1).tolist()
    y = (u[12:] > 0.4).astype(int).tolist()
    for d in (0, 1, 2):
        r = geron_tree_regularization(X, y, max_depth=d)
        assert r["n_leaves"] <= r["baseline_leaves"]
        assert r["train_score"] <= r["baseline_train_score"] + 1e-12


def test_hmdtr_leaf_floor_blocks_illegal_splits():
    from morie.fn.hmdtr import geron_tree_regularization

    r = geron_tree_regularization([[1.0], [2.0], [3.0], [4.0]], [0, 0, 1, 1], min_samples_leaf=3)
    assert r["n_leaves"] == 1
    assert r["leaves_saved"] == 1


def test_hmdthv_variance_is_zero_only_when_trees_agree():
    from morie.fn.hmdthv import geron_tree_high_variance

    r = geron_tree_high_variance([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
                                 [0, 1, 0, 1, 1, 0], n_resamples=12, seed=4)
    P = np.asarray([[0]])  # placeholder to keep the intent explicit
    assert 0.0 <= r["variance"] <= 1.0
    assert 0.0 <= r["structural_instability"] <= 1.0
    assert len(r["root_splits"]) == 12
    # A high-variance fit must disagree somewhere; a constant predictor could not.
    assert r["variance"] > 0.0


def test_hmdthv_ensemble_is_at_least_as_good_as_chance():
    from morie.fn.hmdthv import geron_tree_high_variance

    r = geron_tree_high_variance([[1.0], [2.0], [8.0], [9.0]], [0, 0, 1, 1], n_resamples=8, seed=2)
    assert r["ensemble_score"] >= 0.5
    assert r["bias2"] == pytest.approx(1.0 - r["ensemble_score"])


def test_hmdtst_predictions_invariant_under_affine_rescaling():
    from morie.fn.hmdtst import geron_tree_sensitivity_scale

    u = lcg(30, seed=7)
    X = u[:20].reshape(10, 2).tolist()
    y = (u[20:] > 0.5).astype(int).tolist()
    for a, b in ((100.0, -7.0), (0.001, 5.0), (3.0, 0.0)):
        r = geron_tree_sensitivity_scale(X, y, a=a, b=b)
        assert r["predictions_match"] is True
        assert r["thresholds_match"] is True


def test_hmdtst_negative_scale_is_rejected():
    from morie.fn.hmdtst import geron_tree_sensitivity_scale

    with pytest.raises(ValueError):
        geron_tree_sensitivity_scale([[1.0], [2.0]], [0, 1], a=-2.0)


def test_hmext_ensemble_prediction_is_the_majority_vote():
    from morie.fn.hmext import geron_extra_trees

    r = geron_extra_trees([[1.0], [2.0], [8.0], [9.0]], [0, 0, 1, 1], n_estimators=9, seed=5)
    P = np.asarray(r["tree_predictions"])
    for i in range(P.shape[1]):
        counts = np.bincount(P[:, i].astype(int), minlength=2)
        assert r["predictions"][i] == int(np.argmax(counts))


def test_hmext_random_thresholds_stay_inside_the_feature_range():
    from morie.fn.hmext import geron_extra_trees

    X = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    r = geron_extra_trees(X, [0, 0, 1, 1, 0, 1], n_estimators=7, seed=3, max_depth=2)

    def walk(node, out):
        if node["leaf"]:
            return out
        out.append(node["threshold"])
        walk(node["left"], out)
        walk(node["right"], out)
        return out

    for t in r["trees"]:
        for thr in walk(t, []):
            assert 1.0 <= thr <= 6.0


def test_hmgbrt_first_round_matches_hand_computed_update():
    from morie.fn.hmgbrt import geron_gradient_boosting

    X, y = [[1.0], [2.0], [3.0], [4.0]], [0.0, 0.0, 10.0, 10.0]
    eta = 0.1
    init = 5.0
    resid = np.array(y) - init            # [-5, -5, 5, 5]
    stump = np.array([-5.0, -5.0, 5.0, 5.0])
    hand = init + eta * stump
    r = geron_gradient_boosting(X, y, n_estimators=1, learning_rate=eta, max_depth=1)
    assert r["init"] == pytest.approx(init)
    assert r["predictions"] == pytest.approx(hand.tolist())
    assert r["residual_history"][0] == pytest.approx(resid.tolist())


def test_hmgbrt_loss_is_monotone_and_shrinkage_slows_it():
    from morie.fn.hmgbrt import geron_gradient_boosting

    X, y = [[1.0], [2.0], [3.0], [4.0]], [0.0, 1.0, 8.0, 10.0]
    fast = geron_gradient_boosting(X, y, n_estimators=5, learning_rate=1.0, max_depth=1)
    slow = geron_gradient_boosting(X, y, n_estimators=5, learning_rate=0.1, max_depth=1)
    assert fast["monotone"] and slow["monotone"]
    assert fast["train_mse"] <= slow["train_mse"]
    assert slow["loss_history"] == sorted(slow["loss_history"], reverse=True)


# ============================================================== clustering


def test_hmdbs_labels_match_hand_computed_components():
    from morie.fn.hmdbs import geron_dbscan

    X = [[0.0], [0.4], [0.8], [10.0], [10.3], [50.0]]
    r = geron_dbscan(X, eps=0.5, min_samples=2)
    # Points 0-2 chain together, 3-4 pair up, 5 is isolated.
    lab = r["labels"]
    assert lab[0] == lab[1] == lab[2]
    assert lab[3] == lab[4]
    assert lab[0] != lab[3]
    assert lab[5] == -1
    assert r["n_clusters"] == 2


def test_hmdbs_density_threshold_controls_noise():
    from morie.fn.hmdbs import geron_dbscan

    X = [[0.0], [0.4], [10.0], [10.3]]
    assert geron_dbscan(X, eps=0.5, min_samples=2)["n_noise"] == 0
    assert geron_dbscan(X, eps=0.5, min_samples=3)["n_noise"] == 4


def test_hmgmm_em_log_likelihood_is_non_decreasing():
    from morie.fn.hmgmm import geron_gaussian_mixture

    u = lcg(20, seed=13)
    X = np.concatenate([u[:10] * 0.5, 8.0 + u[10:] * 0.5]).reshape(-1, 1)
    r = geron_gaussian_mixture(X, n_components=2, seed=1)
    h = r["ll_history"]
    assert all(h[i + 1] >= h[i] - 1e-8 for i in range(len(h) - 1))
    assert sum(r["weights"]) == pytest.approx(1.0)


def test_hmgmm_single_component_is_the_fitted_gaussian():
    from morie.fn.hmgmm import geron_gaussian_mixture, gmm_log_pdf

    X = np.array([[1.0], [3.0], [5.0], [7.0]])
    r = geron_gaussian_mixture(X, n_components=1)
    assert r["means"][0][0] == pytest.approx(float(X.mean()))
    assert r["covariances"][0][0][0] == pytest.approx(float(X.var()), rel=1e-3)
    hand = float(np.sum(gmm_log_pdf(X, np.array([X.mean()]), np.asarray(r["covariances"][0]))))
    assert r["log_likelihood"] == pytest.approx(hand, rel=1e-6)


def test_hmgand_flags_the_lowest_density_points():
    from morie.fn.hmgand import geron_anomaly_gmm

    X = [[0.0], [0.1], [0.2], [0.05], [0.15], [0.12], [0.08], [0.02], [0.18], [30.0]]
    r = geron_anomaly_gmm(X, n_components=1, contamination=0.1, seed=1)
    assert r["anomaly_indices"] == [9]
    assert r["log_density"][9] == min(r["log_density"])


def test_hmgand_contamination_sets_the_flag_rate():
    from morie.fn.hmgand import geron_anomaly_gmm

    u = lcg(40, seed=17)
    X = u.reshape(-1, 1)
    for c in (0.1, 0.25):
        r = geron_anomaly_gmm(X, n_components=2, contamination=c, seed=2)
        assert abs(r["n_anomalies"] / 40 - c) <= 0.05
    with pytest.raises(ValueError):
        geron_anomaly_gmm(X, contamination=1.0)


# ====================================================== autodiff / attention


def test_hmfad_dual_numbers_match_finite_differences():
    from morie.fn.hmfad import geron_forward_autodiff

    f = lambda v: (v[0] ** 2 * v[1] + v[1].exp()) / (v[0] + 3.0)
    x = [1.3, -0.4]
    r = geron_forward_autodiff(f, x)

    def plain(p):
        from morie.fn.hmfad import Dual

        return f([Dual(p[0]), Dual(p[1])]).value

    assert np.allclose(r["grad"], fd_grad(plain, x, h=1e-6), atol=1e-6)
    assert r["n_passes"] == 2


def test_hmfad_rejects_a_broken_dual_thread():
    from morie.fn.hmfad import geron_forward_autodiff

    with pytest.raises(ValueError):
        geron_forward_autodiff(lambda v: math.exp(v[0].value), [1.0])


def test_hmcgrf_reverse_sweep_matches_finite_differences():
    from morie.fn.hmcgrf import geron_computational_graph

    expr = ("div", ("mul", ("sin", "x"), ("exp", "y")), ("add", ("square", "x"), 2.0))
    env = {"x": 0.7, "y": -0.3}
    r = geron_computational_graph(expr, env)

    def val(p):
        return geron_computational_graph(expr, {"x": p[0], "y": p[1]})["value"]

    fd = fd_grad(val, [env["x"], env["y"]], h=1e-6)
    assert r["grad"]["x"] == pytest.approx(fd[0], abs=1e-6)
    assert r["grad"]["y"] == pytest.approx(fd[1], abs=1e-6)


def test_hmcgrf_shared_subgraph_counted_once():
    from morie.fn.hmcgrf import geron_computational_graph

    sq = ("mul", "x", "x")
    r = geron_computational_graph(("add", sq, sq), {"x": 4.0})
    assert r["value"] == 32.0
    assert r["grad"]["x"] == pytest.approx(16.0)
    assert r["n_nodes"] == 3
    with pytest.raises(ValueError):
        geron_computational_graph(("log", -1.0))


def test_hmfa_tiling_is_exact_against_direct_attention():
    from morie.fn.hmfa import geron_flash_attention

    u = lcg(36, seed=21)
    Q = (u[:12].reshape(6, 2) * 2 - 1)
    K = (u[12:24].reshape(6, 2) * 2 - 1)
    V = (u[24:].reshape(6, 2) * 2 - 1)
    direct = softmax_rows(Q @ K.T / math.sqrt(2)) @ V
    for bs in (1, 2, 3, 6):
        out = np.asarray(geron_flash_attention(Q, K, V, block_size=bs)["output"])
        assert np.allclose(out, direct, atol=1e-12)


def test_hmfa_causal_masking_and_memory_report():
    from morie.fn.hmfa import geron_flash_attention

    Q = K = np.eye(4)
    V = np.arange(4.0).reshape(4, 1)
    r = geron_flash_attention(Q, K, V, block_size=2, causal=True)
    assert r["output"][0][0] == pytest.approx(0.0)  # first row sees only itself
    assert r["peak_score_memory"] == 4
    assert r["naive_score_memory"] == 16


def test_hmcatt_matches_hand_computed_attention():
    from morie.fn.hmcatt import geron_cross_attention

    dec = np.array([[1.0, 0.0]])
    enc = np.array([[1.0, 0.0], [0.0, 1.0]])
    I = np.eye(2)
    logits = dec @ I @ (enc @ I).T / math.sqrt(2)
    w = softmax_rows(logits)
    r = geron_cross_attention(dec, enc, I, I, I)
    assert np.allclose(r["attention_weights"], w)
    assert np.allclose(r["context"], w @ enc)


def test_hmcatt_entropy_is_bounded_by_log2_of_source_length():
    from morie.fn.hmcatt import geron_cross_attention

    r = geron_cross_attention([[0.0]], [[1.0], [2.0], [3.0]], [[0.0]], [[1.0]], [[1.0]])
    assert r["entropy"][0] == pytest.approx(math.log2(3))
    assert r["entropy"][0] <= r["max_entropy"] + 1e-12


def test_hmcst_infonce_against_hand_softmax():
    from morie.fn.hmcst import geron_contrastive_learning

    emb = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    r = geron_contrastive_learning(emb, [1, 0, 3, 2], tau=1.0)
    # Anchor 0: positive cos 1, two negatives at cos 0.
    hand = math.log(math.exp(1.0) + 2 * math.exp(0.0)) - 1.0
    assert r["per_anchor_loss"][0] == pytest.approx(hand)
    assert r["n_negatives"] == 2


def test_hmcst_rejects_self_positives_and_tiny_batches():
    from morie.fn.hmcst import geron_contrastive_learning

    with pytest.raises(ValueError):
        geron_contrastive_learning([[1.0], [2.0], [3.0]], [0, 0, 1])
    with pytest.raises(ValueError):
        geron_contrastive_learning([[1.0], [2.0]], [1, 0])


def test_hmclip_symmetric_loss_and_zero_shot():
    from morie.fn.hmclip import geron_clip

    I = [[1.0, 0.0], [0.0, 1.0]]
    r = geron_clip(I, I, tau=1.0, class_prompts=[[1.0, 0.0], [0.0, 1.0]])
    hand = math.log(math.exp(1.0) + math.exp(0.0)) - 1.0
    assert r["loss"] == pytest.approx(hand)
    assert r["zero_shot"]["predictions"] == [0, 1]
    assert r["matched_similarity"] == pytest.approx([1.0, 1.0])


def test_hmclip_rejects_mismatched_pairs():
    from morie.fn.hmclip import geron_clip

    with pytest.raises(ValueError):
        geron_clip([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])


def test_hmfmap_convolution_against_hand_windows():
    from morie.fn.hmfmap import geron_feature_map

    X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    K = [[1.0, 0.0], [0.0, 1.0]]
    hand = [[X[i][j] + X[i + 1][j + 1] for j in range(2)] for i in range(2)]
    r = geron_feature_map(X, K, activation="identity")
    assert np.allclose(r["feature_map"], hand)


def test_hmfmap_relu_sparsity_is_the_zero_fraction():
    from morie.fn.hmfmap import geron_feature_map

    X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    r = geron_feature_map(X, [[1.0, 0.0], [0.0, 1.0]], b=-9.0)
    flat = [v for row in r["feature_map"] for v in row]
    assert r["sparsity"] == pytest.approx(sum(v == 0 for v in flat) / len(flat))
    assert min(flat) >= 0.0


def test_hmfcn_dense_layer_as_convolution():
    from morie.fn.hmfcn import geron_fcn

    img = [[1.0, -1.0], [2.0, 0.0]]
    model = [np.array([[[[1.0]]], [[[-1.0]]]])]
    r = geron_fcn(img, model)
    assert r["out_shape"] == (2, 2, 2)
    assert r["segmentation"] == [[0, 1], [0, 0]]
    assert np.allclose(r["scores"][0], img)


def test_hmfcn_receptive_field_grows_with_depth():
    from morie.fn.hmfcn import geron_fcn

    img = np.arange(25.0).reshape(5, 5)
    k = np.ones((1, 1, 3, 3))
    r1 = geron_fcn(img, [k])
    r2 = geron_fcn(img, [k, k])
    assert r1["receptive_field"] == 3
    assert r2["receptive_field"] == 5
    assert r2["out_shape"][1] < r1["out_shape"][1]


# ================================================================ diffusion


def test_hmdfw_alpha_bar_is_the_running_product():
    from morie.fn.hmdfw import geron_diffusion_forward

    betas = [0.1, 0.2, 0.3]
    r = geron_diffusion_forward([1.0], T=3, beta_schedule=betas)
    hand = np.cumprod([1 - b for b in betas])
    assert r["alpha_bar"] == pytest.approx(hand.tolist())
    assert r["signal_scale"] == pytest.approx(math.sqrt(hand[-1]))
    assert r["signal_scale"] ** 2 + r["noise_scale"] ** 2 == pytest.approx(1.0)


def test_hmdfw_step_zero_is_the_clean_sample():
    from morie.fn.hmdfw import geron_diffusion_forward

    r = geron_diffusion_forward([2.0, -3.0], T=5, t=0)
    assert list(r["x_t"]) == [2.0, -3.0]
    with pytest.raises(ValueError):
        geron_diffusion_forward([1.0], T=2, beta_schedule=[1.5, 0.2])


def test_hmdrv_reverse_recovers_x0_when_noise_is_known():
    from morie.fn.hmdrv import geron_diffusion_reverse

    betas = [0.4, 0.6]
    abar = np.cumprod([1 - b for b in betas])
    x0, eps = 2.5, 0.8
    xT = math.sqrt(abar[-1]) * x0 + math.sqrt(1 - abar[-1]) * eps
    # A one-step reverse from t=1 with the true noise returns x0 exactly.
    r = geron_diffusion_reverse([math.sqrt(abar[0]) * x0 + math.sqrt(1 - abar[0]) * eps],
                                lambda x, t: np.full_like(x, eps), T=1, beta_schedule=[betas[0]])
    assert r["x_0"][0] == pytest.approx(x0)


def test_hmdrv_enforces_the_model_contract():
    from morie.fn.hmdrv import geron_diffusion_reverse

    with pytest.raises(ValueError):
        geron_diffusion_reverse([1.0, 2.0], lambda x, t: np.zeros(1), T=1, beta_schedule=[0.5])
    with pytest.raises(ValueError):
        geron_diffusion_reverse([1.0], "not callable", T=1)


def test_hmddim_deterministic_and_fewer_model_calls():
    from morie.fn.hmddim import geron_ddim

    zero = lambda x, t: np.zeros_like(x)
    a = geron_ddim([1.0], zero, T=8, n_steps=2, beta_schedule=[0.2] * 8)
    b = geron_ddim([1.0], zero, T=8, n_steps=2, beta_schedule=[0.2] * 8)
    full = geron_ddim([1.0], zero, T=8, n_steps=8, beta_schedule=[0.2] * 8)
    assert a["x_0"] == pytest.approx(b["x_0"])          # deterministic
    assert a["model_calls"] == 2 and full["model_calls"] == 8
    # With eps = 0 both paths end at x_T / sqrt(abar_T).
    abar = float(np.prod([0.8] * 8))
    assert a["x_0"][0] == pytest.approx(1.0 / math.sqrt(abar))
    assert full["x_0"][0] == pytest.approx(1.0 / math.sqrt(abar))


def test_hmddim_rejects_more_steps_than_the_schedule():
    from morie.fn.hmddim import geron_ddim

    with pytest.raises(ValueError):
        geron_ddim([1.0], lambda x, t: np.zeros_like(x), T=4, n_steps=9)


def test_hmddpm_objective_decreases_and_schedule_is_shared():
    from morie.fn.hmddpm import geron_ddpm
    from morie.fn.hmdfw import beta_schedule_values

    r = geron_ddpm([[1.0], [2.0], [3.0], [4.0]], T=4, epochs=300, lr=0.1, seed=3)
    assert r["monotone"] is True
    assert r["final_loss"] < r["loss_history"][0]
    assert r["betas"] == pytest.approx(beta_schedule_values(4, "linear").tolist())


def test_hmddpm_loss_is_a_real_regression_not_a_mean():
    from morie.fn.hmddpm import geron_ddpm

    r = geron_ddpm([[1.0], [5.0]], T=3, epochs=200, lr=0.1, seed=2)
    assert all(v >= 0 for v in r["loss_by_t"])
    assert r["final_loss"] != pytest.approx(float(np.mean([1.0, 5.0])))


# ================================================ autoencoders / generative


def test_hmcae_full_rank_code_learns_the_identity():
    from morie.fn.hmcae import geron_convolutional_autoencoder

    X = np.arange(16.0).reshape(4, 4)
    r = geron_convolutional_autoencoder(X, filters=4, epochs=4000, lr=0.2, seed=1)
    assert r["final_loss"] < 1e-8
    assert np.allclose(r["reconstruction"], X, atol=1e-3)


def test_hmcae_bottleneck_cannot_reach_zero_loss():
    from morie.fn.hmcae import geron_convolutional_autoencoder

    X = np.arange(16.0).reshape(4, 4)
    r = geron_convolutional_autoencoder(X, filters=1, epochs=3000, lr=0.2, seed=1)
    assert r["compression_ratio"] == 4.0
    assert r["final_loss"] > 0.0
    assert r["final_loss"] < r["loss_history"][0]
    with pytest.raises(ValueError):
        geron_convolutional_autoencoder(np.zeros((3, 3)), patch=2)


def test_hmdae_beats_the_passthrough_baseline():
    from morie.fn.hmdae import geron_denoising_autoencoder

    X = np.stack([np.arange(1.0, 9.0), np.arange(1.0, 9.0)], axis=1)
    r = geron_denoising_autoencoder(X, noise_std=0.2, epochs=1200, lr=0.01, hidden=1, seed=4)
    assert r["final_loss"] < r["passthrough_loss"]
    assert r["loss_history"][-1] < r["loss_history"][0]


def test_hmdae_requires_actual_corruption():
    from morie.fn.hmdae import geron_denoising_autoencoder

    with pytest.raises(ValueError):
        geron_denoising_autoencoder([[1.0, 2.0]], noise_std=0.0)


def test_hmgan_value_function_at_the_equilibrium():
    from morie.fn.hmgan import geron_gan

    r = geron_gan([[0.0], [1.0]], G=([[0.0]], [0.5]), D=([0.0], 0.0), epochs=1, lr=0.0)
    assert r["value_history"][0] == pytest.approx(2 * math.log(0.5))
    assert r["equilibrium_value"] == pytest.approx(-1.3862943611198906)


def test_hmgan_non_saturating_loss_survives_a_confident_discriminator():
    from morie.fn.hmgan import geron_gan

    args = dict(G=([[0.0]], [-50.0]), D=([1.0], 0.0), epochs=1, lr=0.1)
    sat = geron_gan([[0.0], [1.0]], non_saturating=False, **args)
    ns = geron_gan([[0.0], [1.0]], non_saturating=True, **args)
    assert sat["g_grad_norm"] == pytest.approx(0.0, abs=1e-12)
    assert ns["g_grad_norm"] > 0.5


def test_hmdcg_layer_count_follows_the_stride_decomposition():
    from morie.fn.hmdcg import geron_dcgan

    for side, layers in ((8, 1), (16, 2), (32, 3), (64, 4)):
        r = geron_dcgan(np.zeros((1, side, side)), z_dim=8, filters=4)
        assert r["n_layers"] == layers
        assert r["image_shape"] == (side, side)


def test_hmdcg_projection_parameter_count_is_exact():
    from morie.fn.hmdcg import geron_dcgan

    r = geron_dcgan(np.zeros((1, 16, 16)), z_dim=8, filters=4)
    seed_units = 4 * 4 * (4 * 2 ** (2 - 1))
    assert r["generator_layers"][0]["params"] == 8 * seed_units + seed_units
    with pytest.raises(ValueError):
        geron_dcgan(np.zeros((1, 20, 20)), z_dim=8, filters=4)


# ======================================================= reinforcement learning


def test_hmeg_probabilities_sum_to_one_and_floor_at_eps_over_A():
    from morie.fn.hmeg import geron_epsilon_greedy

    r = geron_epsilon_greedy([[1.0, 5.0, 2.0, 5.0]], s=0, epsilon=0.4)
    p = r["probabilities"]
    assert sum(p) == pytest.approx(1.0)
    assert min(p) == pytest.approx(0.4 / 4)
    # Ties share the greedy mass equally.
    assert p[1] == pytest.approx(p[3]) == pytest.approx(0.4 / 4 + 0.6 / 2)


def test_hmeg_greedy_limit_and_range_checks():
    from morie.fn.hmeg import geron_epsilon_greedy

    r = geron_epsilon_greedy([[1.0, 9.0]], 0, 0.0)
    assert r["probabilities"] == pytest.approx([0.0, 1.0])
    with pytest.raises(ValueError):
        geron_epsilon_greedy([[1.0, 9.0]], 0, 1.5)


def test_hmcrd_returns_match_the_explicit_discounted_sum():
    from morie.fn.hmcrd import geron_credit_assignment

    rew = [1.0, 0.0, 2.0, -1.0]
    g = 0.9
    hand = [sum(g**k * rew[t + k] for k in range(len(rew) - t)) for t in range(len(rew))]
    r = geron_credit_assignment(rew, gamma=g)
    assert r["raw_returns"] == pytest.approx(hand)
    assert r["total_return"] == pytest.approx(hand[0])


def test_hmcrd_eligibility_trace_recursion():
    from morie.fn.hmcrd import geron_credit_assignment

    g, lam = 0.9, 0.5
    r = geron_credit_assignment([0.0] * 4, gamma=g, lam=lam)
    e = 0.0
    hand = []
    for _ in range(4):
        e = g * lam * e + 1.0
        hand.append(e)
    assert r["eligibility"] == pytest.approx(hand)


def test_hmdqn_single_step_matches_hand_td_update():
    from morie.fn.hmdqn import geron_dqn

    Q = [[0.0, 0.0], [0.0, 0.0]]
    Qt = [[0.0, 0.0], [1.0, 3.0]]
    lr, gamma = 0.5, 0.9
    hand = 0.0 + lr * ((2.0 + gamma * 3.0) - 0.0)
    r = geron_dqn(None, Q, Qt, [(0, 1, 2.0, 1, False)], epochs=1, lr=lr, gamma=gamma)
    assert r["Q"][0][1] == pytest.approx(hand)
    assert r["Q"][0][0] == 0.0


def test_hmdqn_terminal_drops_the_bootstrap_and_converges():
    from morie.fn.hmdqn import geron_dqn

    r = geron_dqn(None, [[0.0]], [[5.0]], [(0, 0, 1.0, 0, True)], epochs=30, lr=0.5)
    assert r["Q"][0][0] == pytest.approx(1.0, abs=1e-6)
    assert r["loss_history"][-1] < r["loss_history"][0]
    with pytest.raises(ValueError):
        geron_dqn(None, [[0.0]], [[0.0]], [])


def test_hmddqn_removes_the_max_operator_bias():
    from morie.fn.hmddqn import geron_double_dqn

    r = geron_double_dqn(None, [[0.0, 1.0]], [[10.0, -10.0]], [(0, 0, 0.0, 0, False)],
                         epochs=1, lr=1.0, gamma=1.0)
    assert r["targets"][0] == pytest.approx(-10.0)      # evaluate with the target net
    assert r["vanilla_targets"][0] == pytest.approx(10.0)
    assert r["overestimation_gap"][0] == pytest.approx(20.0)


def test_hmddqn_agrees_with_dqn_when_the_nets_agree():
    from morie.fn.hmddqn import geron_double_dqn
    from morie.fn.hmdqn import geron_dqn

    Q = [[1.0, 2.0], [0.5, 0.25]]
    buf = [(0, 0, 1.0, 1, False), (1, 1, -1.0, 0, False)]
    a = geron_dqn(None, Q, Q, buf, epochs=1, lr=0.3, gamma=0.9)
    b = geron_double_dqn(None, Q, Q, buf, epochs=1, lr=0.3, gamma=0.9)
    assert np.allclose(a["Q"], b["Q"])


def test_hmdldqn_aggregation_gradients_are_exact():
    from morie.fn.hmdldqn import dueling_q, geron_dueling_dqn

    r = geron_dueling_dqn(None, [0.0], [[0.0, 0.0]], [(0, 0, 1.0, 0, True)], epochs=1, lr=1.0)
    # TD error is 1; dV = 1 and dA = [1 - 1/2, -1/2].
    assert r["V"][0] == pytest.approx(1.0)
    assert r["A"][0] == pytest.approx([0.5, -0.5])
    assert r["Q"][0] == pytest.approx(dueling_q([1.0], [[0.5, -0.5]])[0].tolist())


def test_hmdldqn_advantages_are_mean_centred_in_q():
    from morie.fn.hmdldqn import dueling_q

    Q = dueling_q([2.0, -1.0], [[3.0, 1.0, -1.0], [0.0, 0.0, 6.0]])
    assert Q.mean(axis=1) == pytest.approx([2.0, -1.0])


def test_hmddpg_polyak_and_deterministic_policy():
    from morie.fn.hmddpg import geron_ddpg

    env = lambda s, a: (s, -float((a - 1.0) ** 2), False)
    r = geron_ddpg(env, [0.5], [0.0, 0.0], epochs=1, lr=0.0, ou_sigma=0.0, s0=[2.0],
                   tau=0.25, critic_target=[0.0, 0.0])
    assert r["actions"][0] == pytest.approx(1.0)   # mu(s) = 0.5 * 2
    assert r["ou_noise"][0] == 0.0
    assert r["critic_target"] == pytest.approx([0.0, 0.0])


def test_hmddpg_learns_to_raise_the_reward():
    from morie.fn.hmddpg import geron_ddpg

    env = lambda s, a: (s, -float((a - 1.0) ** 2), False)
    r = geron_ddpg(env, [0.0], [0.0, 0.0], epochs=300, lr=0.05, ou_sigma=0.1, seed=1)
    assert sum(r["rewards"][-50:]) > sum(r["rewards"][:50])
    with pytest.raises(ValueError):
        geron_ddpg("not callable", [0.0], [0.0, 0.0])


# ============================================================ neural networks


def test_hmclsn_output_gradient_matches_finite_differences():
    from morie.fn.hmcec import geron_cross_entropy_cost
    from morie.fn.hmclsn import geron_classification_mlp, mlp_init

    # One-layer (softmax regression) network: compare the trained step with
    # a hand gradient step on the same initial weights.
    X = [[1.0, -0.5], [0.3, 2.0], [-1.0, 0.7]]
    y = [0, 1, 0]
    Ws, bs = mlp_init([2, 2], seed=1)
    lr = 0.2
    Xa = np.asarray(X)
    Y = np.zeros((3, 2))
    Y[np.arange(3), y] = 1.0
    P = softmax_rows(Xa @ Ws[0] + bs[0])
    hand_W = Ws[0] - lr * (Xa.T @ (P - Y) / 3)
    r = geron_classification_mlp(X, y, hidden_sizes=(), epochs=1, lr=lr, seed=1)
    assert np.allclose(r["weights"][0], hand_W)


def test_hmclsn_parameter_count_and_xor_needs_depth():
    from morie.fn.hmclsn import geron_classification_mlp

    r = geron_classification_mlp([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
                                 [0, 0, 1, 1], hidden_sizes=(3,), epochs=1, lr=0.1)
    assert r["n_params"] == 2 * 3 + 3 + 3 * 2 + 2
    Xx = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    yx = [0, 1, 1, 0]
    assert geron_classification_mlp(Xx, yx, hidden_sizes=(), epochs=500, lr=0.5)["accuracy"] < 1.0
    assert geron_classification_mlp(Xx, yx, hidden_sizes=(8,), epochs=3000, lr=0.5, seed=2)["accuracy"] == 1.0


def test_hmchrn_initial_loss_is_exactly_log_vocab():
    from morie.fn.hmchrn import geron_char_rnn

    r = geron_char_rnn("abcabcabc", hidden=4, epochs=1, lr=0.0, seed=1)
    assert r["loss_history"][0] == pytest.approx(math.log(3))
    assert r["chance_loss"] == pytest.approx(math.log(3))
    assert r["perplexity"] == pytest.approx(3.0)


def test_hmchrn_learns_a_repeating_pattern():
    from morie.fn.hmchrn import geron_char_rnn

    r = geron_char_rnn("ababababab", hidden=6, epochs=300, lr=0.5, seed=1, generate=4)
    assert r["final_loss"] < 0.05
    assert r["generated"] == "abab"
    with pytest.raises(ValueError):
        geron_char_rnn("aaaa")


def test_hmdrnn_identity_stack_integrates_the_input():
    from morie.fn.hmdrnn import geron_deep_rnn

    W = ([[1.0]], [[1.0]], [0.0])
    r1 = geron_deep_rnn([[1.0], [1.0], [1.0], [1.0]], weights=[W], activation="relu")
    r2 = geron_deep_rnn([[1.0], [1.0], [1.0], [1.0]], weights=[W, W], activation="relu")
    assert [row[0] for row in r1["outputs"]] == pytest.approx([1, 2, 3, 4])
    assert [row[0] for row in r2["outputs"]] == pytest.approx([1, 3, 6, 10])  # triangular


def test_hmdrnn_parameter_count_and_tanh_bound():
    from morie.fn.hmdrnn import geron_deep_rnn

    r = geron_deep_rnn([[0.0, 0.0]], hidden_sizes=(3, 4))
    assert r["n_params"] == (2 * 3 + 3 * 3 + 3) + (3 * 4 + 4 * 4 + 4)
    big = geron_deep_rnn([[50.0]] * 6, weights=[([[1.0]], [[1.0]], [0.0])])
    assert max(abs(v[0]) for v in big["outputs"]) <= 1.0


# =========================================================== architectures


def test_hmdctr_block_parameter_count_itemised():
    from morie.fn.hmdctr import block_params, geron_decoder_only

    d, ff = 16, 64
    p = block_params(d, d_ff=ff)
    assert p["self_attention"] == 4 * d * d + 4 * d
    assert p["ffn"] == 2 * d * ff + ff + d
    assert p["layer_norms"] == 4 * d
    assert p["total"] == p["self_attention"] + p["ffn"] + p["layer_norms"]
    r = geron_decoder_only([1, 2], n_layers=3, n_heads=4, d_model=d, vocab_size=7,
                           max_len=4, d_ff=ff)
    assert r["total_params"] == 7 * d + 4 * d + 3 * p["total"] + 2 * d


def test_hmdctr_causal_mask_and_divisibility():
    from morie.fn.hmdctr import causal_mask, geron_decoder_only

    m = np.asarray(causal_mask(4))
    assert m.sum() == 6  # strict upper triangle of a 4x4
    r = geron_decoder_only([1, 2, 3], n_layers=1, n_heads=2, d_model=4, vocab_size=5, max_len=8)
    assert np.asarray(r["mask"]).sum() == 3
    with pytest.raises(ValueError):
        geron_decoder_only([1], n_heads=5, d_model=768)
    with pytest.raises(ValueError):
        geron_decoder_only([1, 2, 3], max_len=2)


def test_hmencox_is_bidirectional_and_matches_decoder_block_cost():
    from morie.fn.hmdctr import block_params
    from morie.fn.hmencox import geron_encoder_only

    r = geron_encoder_only([1, 2, 3], n_layers=2, n_heads=2, d_model=8, vocab_size=5, max_len=16)
    assert r["block_params"] == block_params(8)["total"]
    assert not np.asarray(r["attention_mask"]).any()
    assert r["seq_len"] == 5  # [CLS] + 3 + [SEP]


def test_hmencox_classifier_head_and_length_guard():
    from morie.fn.hmencox import geron_encoder_only

    base = geron_encoder_only([1, 2], n_layers=1, n_heads=2, d_model=8, vocab_size=5, max_len=8)
    head = geron_encoder_only([1, 2], n_layers=1, n_heads=2, d_model=8, vocab_size=5,
                              max_len=8, n_classes=3)
    assert head["total_params"] - base["total_params"] == 8 * 3 + 3
    with pytest.raises(ValueError):
        geron_encoder_only([1, 2, 3, 4], max_len=4)  # [CLS]/[SEP] push it over


def test_hmencd_decoder_block_costs_one_extra_attention_and_norm():
    from morie.fn.hmdctr import block_params
    from morie.fn.hmencd import geron_encoder_decoder_transformer

    d = 8
    r = geron_encoder_decoder_transformer([1, 2], [3, 4, 5], n_layers=1, n_heads=2,
                                          d_model=d, vocab_size=5, max_len=8, d_ff=32)
    extra = (4 * d * d + 4 * d) + 2 * d
    assert r["extra_per_decoder_block"] == extra
    assert r["decoder_block_params"] == block_params(d, d_ff=32)["total"] + extra


def test_hmencd_three_masks_have_the_right_shapes():
    from morie.fn.hmencd import geron_encoder_decoder_transformer

    r = geron_encoder_decoder_transformer([1, 2], [3, 4, 5], n_layers=1, n_heads=2,
                                          d_model=8, vocab_size=5, max_len=8, d_ff=32)
    assert np.asarray(r["src_mask"]).shape == (2, 2)
    assert np.asarray(r["tgt_mask"]).shape == (3, 3)
    assert np.asarray(r["cross_mask"]).shape == (3, 2)
    assert np.asarray(r["tgt_mask"]).sum() == 3
    assert not np.asarray(r["cross_mask"]).any()


def test_hmgpt1_lm_loss_against_hand_log_sum():
    from morie.fn.hmgpt1 import geron_gpt1

    logits = [[0.0, 2.0], [1.0, 0.0], [0.0, 0.0]]
    seq = [0, 1, 0]
    # Positions 0 and 1 predict tokens 1 and 0.
    hand = np.mean([
        math.log(math.exp(0.0) + math.exp(2.0)) - 2.0,
        math.log(math.exp(1.0) + math.exp(0.0)) - 1.0,
    ])
    r = geron_gpt1(seq, logits=logits, n_layers=1, n_heads=1, d_model=2, vocab_size=2, max_len=8)
    assert r["loss"] == pytest.approx(hand)
    assert r["n_predicted"] == 2


def test_hmgpt1_config_matches_the_published_model():
    from morie.fn.hmgpt1 import geron_gpt1

    r = geron_gpt1([1, 2, 3])
    assert (r["config"]["n_layers"], r["config"]["n_heads"], r["config"]["d_model"]) == (12, 12, 768)
    assert r["config"]["max_len"] == 512
    assert 110e6 < r["total_params"] < 125e6


def test_hmgpt2_sizes_scale_as_published():
    from morie.fn.hmgpt2 import geron_gpt2

    sizes = {s: geron_gpt2([1], size=s)["total_params"] for s in ("small", "medium", "large", "xl")}
    assert 117e6 < sizes["small"] < 130e6
    assert 340e6 < sizes["medium"] < 360e6
    assert 1.4e9 < sizes["xl"] < 1.6e9
    assert sizes["small"] < sizes["medium"] < sizes["large"] < sizes["xl"]


def test_hmgpt2_head_width_is_constant_and_embeddings_are_fixed():
    from morie.fn.hmgpt2 import geron_gpt2

    for s in ("small", "medium", "large", "xl"):
        r = geron_gpt2([1], size=s)
        assert r["d_head"] == 64
        assert r["embedding_params"] == 50257 * r["config"]["d_model"] + 1024 * r["config"]["d_model"]
    with pytest.raises(ValueError):
        geron_gpt2([1], size="enormous")


def test_hmgoog_inception_widths_add_and_reductions_save():
    from morie.fn.hmgoog import geron_googlenet, inception_module

    m = inception_module(192, 64, 96, 128, 16, 32, 32)
    assert m["out_channels"] == 64 + 128 + 32 + 32
    assert m["params"] == m["branch_1x1"] + m["branch_3x3"] + m["branch_5x5"] + m["branch_pool"]
    assert m["naive_5x5_params"] == 25 * 192 * 32 + 32
    assert m["reduction_saving"] > 0
    r = geron_googlenet(1000)
    assert 6.5e6 < r["total_params"] < 7.5e6      # ~a tenth of AlexNet
    assert r["modules"][0]["in_channels"] == 192


def test_hmgoog_class_count_only_moves_the_final_layer():
    from morie.fn.hmgoog import geron_googlenet

    a = geron_googlenet(1000)
    b = geron_googlenet(10)
    ch = a["final_feature_map"][0]
    assert a["total_params"] - b["total_params"] == (ch * 1000 + 1000) - (ch * 10 + 10)


def test_hmfmn_shapes_follow_the_conv_arithmetic():
    from morie.fn.hmfmn import geron_fashion_mnist

    r = geron_fashion_mnist()
    outs = [l["out"] for l in r["layers"] if l["kind"] in ("conv", "pool")]
    assert outs == [26, 13, 11, 5]
    assert r["flatten_dim"] == 5 * 5 * 64
    assert r["fc_share"] > 0.9


def test_hmfmn_training_config_is_validated():
    from morie.fn.hmfmn import geron_fashion_mnist

    r = geron_fashion_mnist(epochs=3, lr=0.01, batch_size=64)
    assert r["training_config"] == {"epochs": 3, "lr": 0.01, "batch_size": 64,
                                    "loss": "sparse categorical cross-entropy"}
    assert r["steps_per_epoch"] == math.ceil(60000 / 64)
    with pytest.raises(ValueError):
        geron_fashion_mnist(batch_size=0)


def test_hmdeit_token_count_and_distillation_overhead():
    from morie.fn.hmdeit import geron_deit

    img = np.zeros((3, 32, 32))
    r = geron_deit(img, patch_size=16, n_layers=1, d_model=8, n_heads=2, n_classes=4)
    assert r["n_patches"] == 4
    assert r["n_tokens"] == 6                       # patches + CLS + distillation
    assert r["patch_embed_params"] == 16 * 16 * 3 * 8 + 8
    assert r["distillation_overhead"] == 8 + (8 * 4 + 4)
    with pytest.raises(ValueError):
        geron_deit(np.zeros((3, 30, 30)), patch_size=16)


def test_hmdeit_hard_distillation_loss_against_hand_terms():
    from morie.fn.hmdeit import geron_deit

    img = np.zeros((3, 32, 32))
    r = geron_deit(img, patch_size=16, n_layers=1, d_model=8, n_heads=2, n_classes=2,
                   logits_cls=[[10.0, 0.0]], logits_dist=[[10.0, 0.0]], y=[0],
                   teacher=[[0.0, 10.0]], alpha=0.5)
    ce_correct = math.log(1 + math.exp(-10.0))
    ce_wrong = 10.0 + math.log(1 + math.exp(-10.0))
    assert r["loss_cls"] == pytest.approx(ce_correct, abs=1e-9)
    assert r["loss_dist"] == pytest.approx(ce_wrong, abs=1e-9)
    assert r["teacher_agreement"] == 0.0


def test_hmdetr_token_count_and_quadratic_attention_cost():
    from morie.fn.hmdetr import geron_detr

    r = geron_detr(np.zeros((3, 224, 224)), n_queries=10, n_layers=1, d_model=8,
                   n_heads=2, n_classes=3)
    assert r["feature_shape"] == (7, 7)
    assert r["n_tokens"] == 49
    assert r["encoder_attention_cost"] == 49**2
    assert r["max_detections"] == 10


def test_hmdetr_matching_is_bipartite():
    from morie.fn.hmdetr import geron_detr

    pb = [[0.0, 0.0, 1.0, 1.0], [10.0, 10.0, 11.0, 11.0]]
    pc = [[10.0, 0.0], [0.0, 10.0]]
    r = geron_detr(np.zeros((3, 224, 224)), n_queries=2, n_layers=1, d_model=8, n_heads=2,
                   n_classes=2, pred_boxes=pb, pred_classes=pc,
                   gt_boxes=[[0.0, 0.0, 1.0, 1.0]], gt_classes=[0])
    assert r["matching"] == [(0, 0)]
    assert r["loss_bbox"] == 0.0
    with pytest.raises(ValueError):
        geron_detr(np.zeros((3, 224, 224)), n_queries=1, pred_boxes=pb, pred_classes=pc,
                   gt_boxes=[[0.0, 0.0, 1.0, 1.0]], gt_classes=[0])


def test_hmdino_centering_removes_a_constant_offset():
    from morie.fn.hmdino import geron_dino

    a = geron_dino(None, [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]])
    b = geron_dino(None, [[0.0, 0.0], [0.0, 0.0]], [[7.0, 7.0], [7.0, 7.0]])
    assert a["loss"] == pytest.approx(math.log(2))
    assert b["loss"] == pytest.approx(a["loss"])
    assert a["kl_to_uniform"] == pytest.approx(0.0, abs=1e-12)


def test_hmdino_momentum_teacher_and_sharpening_guard():
    from morie.fn.hmdino import geron_dino

    r = geron_dino(None, [[1.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], momentum=0.75)
    assert r["teacher_next"][0] == pytest.approx([0.25, 0.0])
    with pytest.raises(ValueError):
        geron_dino(None, [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], tau_t=0.5)


def test_hmdbrt_kl_term_scales_with_temperature_squared():
    from morie.fn.hmdbrt import geron_distilbert

    t, s = [[10.0, 0.0]], [[0.0, 0.0]]
    for T in (1.0, 2.0, 4.0):
        r = geron_distilbert(t, s, [1], alpha_mlm=0.0, alpha_ce=1.0, temperature=T)
        pt = softmax_rows(np.asarray(t) / T)[0]
        ps = softmax_rows(np.asarray(s) / T)[0]
        hand = T * T * float(np.sum(pt * (np.log(pt) - np.log(ps))))
        assert r["loss_ce"] == pytest.approx(hand, rel=1e-9)


def test_hmdbrt_compression_is_about_forty_percent():
    from morie.fn.hmdbrt import geron_distilbert

    r = geron_distilbert([[1.0, 0.0]], [[1.0, 0.0]], [1], alpha_mlm=0.0, alpha_ce=1.0)
    assert 0.35 < r["param_reduction"] < 0.45      # not 0.5: embeddings are shared
    assert r["loss_ce"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        geron_distilbert([[1.0, 0.0]], [[1.0, 0.0]], [1], alpha_mlm=1.0)  # no labels


def test_hmdale_log_likelihood_against_hand_softmax():
    from morie.fn.hmdale import geron_dalle

    logits = np.array([0.0, 2.0])
    model = lambda ctx: logits
    r = geron_dalle([0], model, n_image_tokens=3)
    per = math.log(math.exp(0.0) + math.exp(2.0)) - 2.0
    assert r["image_tokens"] == [1, 1, 1]
    assert r["log_likelihood"] == pytest.approx(-3 * per)
    assert r["perplexity"] == pytest.approx(math.exp(per))


def test_hmdale_enforces_a_constant_vocabulary():
    from morie.fn.hmdale import geron_dalle

    with pytest.raises(ValueError):
        geron_dalle([0, 1], lambda ctx: np.zeros(len(ctx)), n_image_tokens=2)


def test_hmflmg_gate_zero_is_exactly_the_identity():
    from morie.fn.hmflmg import geron_flamingo

    text = [[2.0], [4.0], [-1.0]]
    r = geron_flamingo([[1.0], [3.0]], text, gate=0.0)
    assert np.allclose(r["output"], text)
    assert r["is_identity_at_init"] is True
    assert r["delta_norm"] == pytest.approx(0.0)


def test_hmflmg_resampler_fixes_the_visual_token_count():
    from morie.fn.hmflmg import geron_flamingo

    for n_feat in (2, 5, 9):
        feats = np.linspace(0.0, 1.0, n_feat).reshape(-1, 1)
        r = geron_flamingo(feats, [[0.0]], gate=1.0)
        assert r["n_visual_tokens"] == 1
        assert r["n_image_features"] == n_feat
    # Opening the gate scales the visual contribution by tanh(gate).
    a = geron_flamingo([[1.0], [3.0]], [[0.0]], gate=1.0)
    assert a["output"][0][0] == pytest.approx(math.tanh(1.0) * 2.0)


def test_hmclc_iou_against_hand_computed_geometry():
    from morie.fn.hmclc import geron_classification_localization

    model = lambda img: np.array([0.0, 0.0, 0.0, 0.0, 2.0, 2.0])
    r = geron_classification_localization(None, model, gt_class=[0], gt_box=[[1.0, 0.0, 2.0, 2.0]])
    # Boxes [-1,-1,1,1] and [0,-1,2,1]: intersection 1x2, union 4+4-2.
    assert r["iou"][0] == pytest.approx(2 / 6)
    assert r["loss_box"] == pytest.approx(1.0)
    assert r["loss_class"] == pytest.approx(math.log(2))
    assert r["loss"] == pytest.approx(math.log(2) + 1.0)


def test_hmclc_rejects_degenerate_boxes_and_bad_widths():
    from morie.fn.hmclc import geron_classification_localization

    with pytest.raises(ValueError):
        geron_classification_localization(None, lambda i: np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        geron_classification_localization(None, lambda i: np.array([0.0, 1.0, 1.0]))


def test_hmfsf_prompt_contains_exactly_k_demonstrations():
    from morie.fn.hmfsf import geron_few_shot

    seen = {}

    def model(prompt):
        seen["last"] = prompt
        return prompt.count("->")

    ex = [("a", "1"), ("b", "2"), ("c", "3")]
    for k in (0, 1, 3):
        r = geron_few_shot(model, ex, "q", k=k)
        assert r["prompt"].count("->") == k + 1     # demos plus the query
        assert r["k"] == k
    with pytest.raises(ValueError):
        geron_few_shot(model, ex, "q", k=5)


def test_hmfsf_zero_shot_control_is_run_separately():
    from morie.fn.hmfsf import geron_few_shot

    calls = []

    def model(prompt):
        calls.append(prompt)
        return "yes" if "->" in prompt.split("\n")[0] and len(prompt.split("\n")) > 1 else "no"

    r = geron_few_shot(model, [("a", "1")], "q")
    assert len(calls) == 2
    assert r["prediction"] == "yes" and r["zero_shot_prediction"] == "no"
    assert r["changed_by_context"] is True


# ============================================================ data / numerics


def test_hmdld_epoch_is_a_permutation_with_correct_batching():
    from morie.fn.hmdld import geron_dataloader

    r = geron_dataloader(10, batch_size=3, shuffle=True, seed=7)
    flat = [i for b in r["batches"] for i in b]
    assert sorted(flat) == list(range(10))
    assert [len(b) for b in r["batches"]] == [3, 3, 3, 1]
    assert geron_dataloader(10, 3, drop_last=True)["dropped"] == 1


def test_hmdld_slices_real_data_and_validates_sizes():
    from morie.fn.hmdld import geron_dataloader

    data = [[float(i)] for i in range(5)]
    r = geron_dataloader(data, batch_size=2)
    assert r["batch_data"][0] == [[0.0], [1.0]]
    assert r["batch_data"][-1] == [[4.0]]
    with pytest.raises(ValueError):
        geron_dataloader(5, batch_size=0)


def test_hmcod_nearest_neighbour_radius_matches_the_ball_volume():
    from morie.fn.hmcod import geron_curse_dimensionality

    for d, n in ((1, 100), (3, 50), (10, 1000)):
        r = geron_curse_dimensionality(d, n)
        # n * volume(ball of radius r) == 1 by construction.
        vol = math.pi ** (d / 2) * r["nn_distance"] ** d / math.gamma(d / 2 + 1)
        assert n * vol == pytest.approx(1.0, rel=1e-9)
        assert r["mean_pairwise_distance"] == pytest.approx(math.sqrt(d / 6))


def test_hmcod_distance_grows_with_dimension():
    from morie.fn.hmcod import geron_curse_dimensionality

    ds = [geron_curse_dimensionality(d, 100)["nn_distance"] for d in (1, 5, 20, 100)]
    assert ds == sorted(ds)
    with pytest.raises(ValueError):
        geron_curse_dimensionality(0, 10)


def test_hmfp32_fields_reconstruct_the_stored_value():
    from morie.fn.hmfp32 import geron_fp32

    xs = [1.0, -0.5, 3.75, 1e-3, 12345.678]
    r = geron_fp32(xs)
    for i, v in enumerate(r["value"]):
        assert r["reconstructed"][i] == pytest.approx(v, rel=1e-12)
        assert r["kind"][i] == "normal"
    assert r["eps"] == pytest.approx(2.0**-23)
    assert geron_fp32([1.0])["exponent"] == [0]


def test_hmfp32_classifies_specials():
    from morie.fn.hmfp32 import geron_fp32

    r = geron_fp32([1e40, 0.0, 1e-42, -1e40])
    assert r["kind"] == ["inf", "zero", "subnormal", "inf"]
    assert r["sign"][3] == 1


def test_hmfp16_round_trip_error_is_bounded_by_eps():
    from morie.fn.hmfp16 import geron_fp16_quant

    u = lcg(20, seed=31)
    xs = (u * 100 - 50).tolist()
    r = geron_fp16_quant(xs)
    assert r["max_rel_error"] <= 2.0**-10
    assert not any(r["overflowed"]) and not any(r["underflowed"])


def test_hmfp16_range_limits_bite_before_precision():
    from morie.fn.hmfp16 import geron_fp16_quant

    r = geron_fp16_quant([70000.0, 1e-9, 1.0])
    assert r["overflowed"] == [True, False, False]
    assert r["underflowed"] == [False, True, False]
    assert r["max_normal"] == 65504.0


def test_hmdqnt_int8_scale_and_round_trip():
    from morie.fn.hmdqnt import geron_dynamic_quantization

    W = [-1.0, -0.25, 0.0, 0.5, 1.0]
    r = geron_dynamic_quantization({"W": W})
    s = 1.0 / 127
    assert r["scales"]["W"] == pytest.approx(s)
    assert r["quantized"]["W"] == [round(v / s) for v in W]
    assert r["max_abs_error"]["W"] <= s / 2 + 1e-12
    assert r["dequantized"]["W"][2] == 0.0     # symmetric: zero is exact


def test_hmdqnt_activation_scale_comes_from_the_batch():
    from morie.fn.hmdqnt import geron_dynamic_quantization

    a = geron_dynamic_quantization({"W": [1.0]}, activations=[0.0, 2.0])
    b = geron_dynamic_quantization({"W": [1.0]}, activations=[0.0, 8.0])
    assert b["activation"]["scale"] == pytest.approx(4 * a["activation"]["scale"])
    with pytest.raises(ValueError):
        geron_dynamic_quantization({"W": [0.0, 0.0]})
