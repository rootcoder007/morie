# morie.fn -- test file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent-route checks for the w4d tranche (Géron modules).

Every assertion here derives the truth by a *different* route than the
implementation: central finite differences for gradients, brute-force
counting for metrics, hand log-sums for losses, explicit refits for
leave-one-out, exhaustive enumeration for dynamic programs, and
monotonicity/bounds where an exact value is not available.

Nothing uses a random number generator: data are explicit lists or come
from the LCG ``s = (1664525*s + 1013904223) % 2**32; u = (s + 0.5)/2**32``.
"""

import math
import os
import tempfile
from collections import Counter
from itertools import combinations

from morie.fn import _array_core as np
import pytest

# -- modules under test -------------------------------------------------
from morie.fn.hmrvn import geron_revnet
from morie.fn.hmrwd import geron_reward_function
from morie.fn.hmsac import geron_sac
from morie.fn.hmsae import _backward, _forward, geron_stacked_autoencoder
from morie.fn.hmsatt import geron_self_attention
from morie.fn.hmsdp import geron_scaled_dot_product
from morie.fn.hmself import geron_self_supervised
from morie.fn.hmselu import geron_selu
from morie.fn.hmsem import geron_semisupervised
from morie.fn.hmsenet import geron_senet
from morie.fn.hmsent import geron_sentiment_analysis
from morie.fn.hmseq2 import geron_seq2seq
from morie.fn.hmsft import geron_sft
from morie.fn.hmsftm import geron_softmax_function
from morie.fn.hmsfts import geron_softmax_score
from morie.fn.hmsgdc import geron_sgd_classifier
from morie.fn.hmsgdu import geron_sgd_update
from morie.fn.hmsigm import geron_sigmoid
from morie.fn.hmsil import geron_silhouette
from morie.fn.hmspcl import geron_spectral_clustering
from morie.fn.hmsrnn import geron_simple_rnn
from morie.fn.hmsrp import geron_sparse_rand_projection
from morie.fn.hmsslc import geron_semisupervised_cluster
from morie.fn.hmssg import geron_semantic_segmentation
from morie.fn.hmstk import geron_stacking
from morie.fn.hmstr import geron_stratified_sampling
from morie.fn.hmstr2 import geron_stride
from morie.fn.hmstz import geron_standardization
from morie.fn.hmsup import geron_supervised_learning
from morie.fn.hmsvdp import geron_svd_pseudoinverse
from morie.fn.hmsvm2 import geron_save_load_pytorch
from morie.fn.hmswi import geron_swish
from morie.fn.hmswin import geron_swin
from morie.fn.hmsymd import evaluate, geron_symbolic_diff, parse
from morie.fn.hmt5 import geron_t5, restore
from morie.fn.hmtanh import geron_tanh
from morie.fn.hmtcmp import geron_torch_compile, matmul_order
from morie.fn.hmtd import geron_td_learning
from morie.fn.hmtd3 import geron_td3
from morie.fn.hmtfl import geron_transfer_learning
from morie.fn.hmtfm import encoder_params, geron_transformer
from morie.fn.hmtlu import geron_tlu
from morie.fn.hmtpp import geron_tensor_parallelism
from morie.fn.hmtrlf import geron_trl_finetune
from morie.fn.hmtsc import geron_torchscript, run_graph
from morie.fn.hmtsf import geron_time_series_forecast
from morie.fn.hmtsne import geron_tsne
from morie.fn.hmuf import geron_underfitting
from morie.fn.hmumap import geron_umap
from morie.fn.hmuns import geron_unsupervised_learning
from morie.fn.hmunsp import geron_unsupervised_pretraining
from morie.fn.hmvae import geron_vae, vae_loss_and_grads
from morie.fn.hmvbgm import digamma, geron_variational_bayes_gmm
from morie.fn.hmvbrt import geron_videobert
from morie.fn.hmvf import geron_value_function
from morie.fn.hmvgr import geron_vanishing_gradients
from morie.fn.hmvilb import geron_vilbert
from morie.fn.hmvit import geron_vision_transformer
from morie.fn.hmvqv import geron_vq_vae, quantize
from morie.fn.hmvth import geron_voting_hard
from morie.fn.hmwemb import geron_word_embeddings
from morie.fn.hmwpt import geron_wordpiece_tokenizer
from morie.fn.hmwrst import geron_warm_restarts
from morie.fn.hmxav import geron_glorot_init
from morie.fn.hmxcpt import geron_xception, separable_params
from morie.fn.hmxgb import geron_xgboost
from morie.fn.hmxgr import geron_exploding_gradients
from morie.fn.hmxln import geron_xlnet, permutation_masks
from morie.fn.hmyolo import box_iou, geron_yolo
from morie.fn.hmzsl import geron_zero_shot

TOL = 1e-6


# -- shared helpers -----------------------------------------------------
def lcg(n, seed=1):
    """The tranche's reference LCG stream on (0, 1)."""
    s = int(seed) % 2**32
    out = []
    for _ in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out.append((s + 0.5) / 2**32)
    return out


def central(f, x, h=1e-6):
    """Scalar central finite difference."""
    return (f(x + h) - f(x - h)) / (2 * h)


def grad_fd(f, x, h=1e-6):
    """Vector central finite difference of a scalar function."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        up, dn = x.copy(), x.copy()
        up[i] += h
        dn[i] -= h
        g[i] = (f(up) - f(dn)) / (2 * h)
    return g


# =======================================================================
# activations and elementary maps
# =======================================================================
def test_hmsigm_derivative_matches_finite_difference():
    f = lambda t: float(geron_sigmoid([t])["a"][0])
    for t in (-2.0, -0.3, 0.0, 1.7):
        assert geron_sigmoid([t])["grad"][0] == pytest.approx(central(f, t), abs=1e-7)


def test_hmsigm_reflection_identity():
    # sigma(x) + sigma(-x) = 1 -- a route the implementation never uses.
    for t in (-5.0, -0.5, 0.0, 3.0):
        total = float(geron_sigmoid([t])["a"][0]) + float(geron_sigmoid([-t])["a"][0])
        assert total == pytest.approx(1.0, abs=1e-12)


def test_hmtanh_against_exponential_definition():
    for z in (-1.5, 0.0, 0.25, 2.0):
        want = (math.exp(z) - math.exp(-z)) / (math.exp(z) + math.exp(-z))
        got = geron_tanh([z])
        assert float(got["a"][0]) == pytest.approx(want, abs=1e-12)
        assert float(got["grad"][0]) == pytest.approx(central(lambda u: math.tanh(u), z), abs=1e-7)


def test_hmselu_continuity_and_slope():
    r = geron_selu([-1e-9, 1e-9])
    assert float(r["a"][0]) == pytest.approx(float(r["a"][1]), abs=1e-8)
    f = lambda z: float(geron_selu([z])["a"][0])
    for z in (-1.3, -0.2, 0.4, 2.0):
        assert float(geron_selu([z])["grad"][0]) == pytest.approx(central(f, z), abs=1e-6)


def test_hmswi_gradient_and_nonmonotonicity():
    f = lambda z: float(geron_swish([z])["a"][0])
    for z in (-3.0, -1.0, 0.0, 2.0):
        assert float(geron_swish([z])["grad"][0]) == pytest.approx(central(f, z), abs=1e-6)
    # swish dips below zero on the negative side; a monotone activation cannot.
    grid = np.linspace(-4, 0, 41)
    vals = geron_swish(grid)["a"]
    assert float(np.min(vals)) < -0.2


def test_hmsftm_against_explicit_exponentials():
    scores = [0.5, -1.25, 2.0]
    denom = sum(math.exp(s) for s in scores)
    want = [math.exp(s) / denom for s in scores]
    p = geron_softmax_function(scores)["p"]
    assert [float(v) for v in p] == pytest.approx(want, abs=1e-12)
    shifted = geron_softmax_function([s + 100.0 for s in scores])["p"]
    assert [float(v) for v in shifted] == pytest.approx(want, abs=1e-12)


def test_hmsftm_jacobian_matches_finite_difference():
    scores = np.array([0.3, -0.7, 1.1])
    J = geron_softmax_function(scores)["jacobian"]
    for k in range(3):
        fd = grad_fd(lambda s: float(geron_softmax_function(s)["p"][k]), scores)
        assert np.allclose(J[k], fd, atol=1e-6)


def test_hmsfts_scores_are_manual_dot_products():
    X = [[1.0, 2.0], [0.5, -1.0]]
    theta = [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]]
    r = geron_softmax_score(X, theta)
    for i, row in enumerate(X):
        for k in range(3):
            want = sum(row[j] * theta[j][k] for j in range(2))
            assert float(r["scores"][i, k]) == pytest.approx(want, abs=1e-12)
    assert list(r["predicted"]) == [int(np.argmax(r["scores"][i])) for i in range(2)]


def test_hmtlu_reproduces_or_gate_truth_table():
    rows = [[0, 0], [0, 1], [1, 0], [1, 1]]
    r = geron_tlu(rows, [1.0, 1.0], -0.5)
    assert [int(v) for v in r["y"]] == [0, 1, 1, 1]
    # brute force: fire iff w.x + b >= 0
    assert [int(sum(x) - 0.5 >= 0) for x in rows] == [int(v) for v in r["y"]]


def test_hmstz_matches_statistics_module():
    col = [4.0, 8.0, 15.0, 16.0, 23.0, 42.0]
    mu = sum(col) / len(col)
    sd = math.sqrt(sum((c - mu) ** 2 for c in col) / len(col))
    z = geron_standardization(col)["X_std"].ravel()
    assert [float(v) for v in z] == pytest.approx([(c - mu) / sd for c in col], abs=1e-12)
    with pytest.raises(ValueError):
        geron_standardization([[1.0, 3.0], [1.0, 5.0]])


def test_hmstr2_matches_brute_force_window_count():
    for in_dim, k, p, s in ((28, 3, 1, 1), (227, 11, 0, 4), (10, 4, 2, 3)):
        starts = [i for i in range(0, in_dim + 2 * p - k + 1, s)]
        assert int(geron_stride(in_dim, k, p, s)["output_dim"]) == len(starts)
    with pytest.raises(ValueError):
        geron_stride(5, 9)


def test_hmxav_variance_matches_target():
    r = geron_glorot_init(40, 60, seed=3)
    W = r["W"]
    assert W.shape == (40, 60)
    assert float(np.var(W)) == pytest.approx(2.0 / 100, rel=0.15)
    assert float(np.max(np.abs(W))) <= float(r["limit"]) + 1e-12
    assert float(r["limit"]) == pytest.approx(math.sqrt(6.0 / 100), abs=1e-12)


# =======================================================================
# attention / transformer family
# =======================================================================
def test_hmsdp_matches_hand_rolled_attention():
    Q = [[1.0, 0.5]]
    K = [[1.0, 0.0], [0.0, 2.0]]
    V = [[1.0, 0.0], [0.0, 1.0]]
    scores = [sum(q * k for q, k in zip(Q[0], row)) / math.sqrt(2) for row in K]
    denom = sum(math.exp(s) for s in scores)
    want_a = [math.exp(s) / denom for s in scores]
    want_y = [want_a[0] * 1.0, want_a[1] * 1.0]
    r = geron_scaled_dot_product(Q, K, V)
    assert [float(v) for v in r["attention"][0]] == pytest.approx(want_a, abs=1e-12)
    assert [float(v) for v in r["Y"][0]] == pytest.approx(want_y, abs=1e-12)


def test_hmsdp_causal_mask_blocks_the_future():
    Q = K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    V = [[1.0], [10.0], [100.0]]
    mask = np.tril(np.ones((3, 3)))
    r = geron_scaled_dot_product(Q, K, V, mask=mask)
    A = r["attention"]
    assert float(A[0, 1]) == 0.0 and float(A[0, 2]) == 0.0
    assert float(A[1, 2]) == 0.0
    assert [float(v) for v in A.sum(axis=1)] == pytest.approx([1.0, 1.0, 1.0], abs=1e-12)
    with pytest.raises(ValueError):
        geron_scaled_dot_product(Q, K, V, mask=np.zeros((3, 3)))


def test_hmsatt_equals_manual_projection_into_hmsdp():
    X = np.array([[1.0, 2.0], [0.0, -1.0]])
    Wq = np.array([[1.0, 0.0], [0.5, 1.0]])
    Wk = np.array([[0.0, 1.0], [1.0, 0.0]])
    Wv = np.array([[2.0, 0.0], [0.0, 3.0]])
    ref = geron_scaled_dot_product(X @ Wq, X @ Wk, X @ Wv, d_k=2)
    got = geron_self_attention(X, Wq, Wk, Wv)
    assert np.allclose(got["Y"], ref["Y"], atol=1e-12)
    with pytest.raises(ValueError):
        geron_self_attention(X, np.ones((3, 2)), Wk, Wv)


def test_hmtfm_parameter_count_recomputed_by_hand():
    d, ff, L = 8, 32, 3
    want = L * (4 * d * d + d * ff + ff + ff * d + d + 4 * d)
    assert encoder_params(d, ff, L) == want
    X = np.eye(4, 8)
    r = geron_transformer(X, n_heads=2, n_layers=1)
    assert int(r["total_params"]) == encoder_params(8, 32, 1)
    # post-norm output rows are standardised
    assert np.allclose(r["Y"].mean(axis=1), 0.0, atol=1e-10)
    assert np.allclose(r["Y"].std(axis=1), 1.0, atol=1e-4)


def test_hmvit_patch_count_and_parameters():
    img = [[float(i * 6 + j) for j in range(6)] for i in range(6)]
    r = geron_vision_transformer(img, patch_size=3, n_layers=1, d_model=4, n_heads=2, n_classes=5)
    assert int(r["n_patches"]) == (6 // 3) ** 2
    assert int(r["patch_dim"]) == 3 * 3 * 1
    assert int(r["seq_len"]) == r["n_patches"] + 1
    hand = 9 * 4 + 4 + 4 + r["seq_len"] * 4 + encoder_params(4, 16, 1) + 4 * 5 + 5
    assert int(r["total_params"]) == hand
    assert r["logits"].shape == (5,)


def test_hmswin_windows_are_local_without_shift():
    img = [[float(v) for v in row] for row in ([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16])]
    base = geron_swin(img, window_size=2, n_layers=1, d_model=3)["Y"]
    bumped = [row[:] for row in img]
    bumped[3][3] += 100.0  # a pixel in the opposite window
    other = geron_swin(bumped, window_size=2, n_layers=1, d_model=3)["Y"]
    assert np.allclose(base[:2, :2], other[:2, :2], atol=1e-12)
    assert not np.allclose(base[2:, 2:], other[2:, 2:], atol=1e-9)


def test_hmrvn_inverts_a_nonlinear_block():
    u = lcg(4, seed=11)
    x = np.array(u) * 4 - 2
    F = lambda a: np.tanh(3 * a) - 0.5
    G = lambda a: np.sin(a) * 2
    r = geron_revnet(x, F, G)
    y1 = x[:2] + F(x[2:])
    y2 = x[2:] + G(y1)
    assert np.allclose(r["y"], np.concatenate([y1, y2]), atol=1e-12)
    assert float(r["reconstruction_error"]) < 1e-12
    with pytest.raises(ValueError):
        geron_revnet(x, lambda a: a[:1], G)


def test_hmsenet_gate_is_sigmoid_of_bottleneck():
    x = np.array([[[2.0, 6.0], [4.0, 10.0]]])  # (1, 2, 2) -> C = 2
    W1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    W2 = np.array([[0.5, 0.0], [0.0, -0.5]])
    r = geron_senet(x, r=1, W1=W1, W2=W2)
    z = x.mean(axis=(0, 1))
    h = np.maximum(z @ W1, 0.0)
    want = 1.0 / (1.0 + np.exp(-(h @ W2)))
    assert np.allclose(r["s"], want, atol=1e-12)
    assert np.allclose(r["y"], x * want, atol=1e-12)


def test_hmxcpt_separable_saving_and_totals():
    assert separable_params(3, 728, 728) == 9 * 728 + 728 * 728
    dense = 3 * 3 * 728 * 728
    assert separable_params(3, 728, 728) / dense == pytest.approx(1 / 728 + 1 / 9, rel=1e-9)
    r = geron_xception(1000)
    assert int(r["trainable_params"]) == int(r["weight_params"]) + 2 * int(r["bn_channels"])
    assert int(r["total_params"]) == int(r["trainable_params"]) + int(r["non_trainable_params"])
    assert int(r["n_separable"]) == sum(1 for l in r["layers"] if l["kind"] == "separable3x3")


# =======================================================================
# sequence models
# =======================================================================
def test_hmsrnn_matches_a_hand_unrolled_loop():
    X = [[0.5], [-1.0], [2.0]]
    Wx, Wh, b = [[2.0]], [[0.5]], [0.25]
    h = 0.0
    want = []
    for x in X:
        h = math.tanh(2.0 * x[0] + 0.5 * h + 0.25)
        want.append(h)
    r = geron_simple_rnn(X, Wx, Wh, b)
    assert [float(v) for v in r["H"].ravel()] == pytest.approx(want, abs=1e-12)
    assert float(r["h_T"][0]) == pytest.approx(want[-1], abs=1e-12)


def test_hmtsf_extrapolates_and_scores_persistence():
    y = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    r = geron_time_series_forecast(y, horizon=2, window=2)
    assert [float(v) for v in r["forecast"]] == pytest.approx([16.0, 18.0], abs=1e-8)
    naive = sum((y[i] - y[i - 1]) ** 2 for i in range(2, len(y))) / (len(y) - 2)
    assert float(r["naive_mse"]) == pytest.approx(naive, abs=1e-12)
    with pytest.raises(ValueError):
        geron_time_series_forecast([1.0, 2.0], horizon=1, window=5)


def test_hmseq2_loss_matches_hand_log_sum():
    z = lambda s: np.asarray([1.0])
    dec = lambda zz, prefix: np.asarray([0.0, math.log(3.0), 0.0])
    # softmax over (0, log 3, 0) is (1/5, 3/5, 1/5)
    r = geron_seq2seq([1], [1, 0], z, dec)
    want = -(math.log(0.6) + math.log(0.2)) / 2
    assert float(r["loss"]) == pytest.approx(want, abs=1e-12)
    assert float(r["perplexity"]) == pytest.approx(math.exp(want), abs=1e-12)
    assert [int(v) for v in r["greedy"]] == [1, 1]


def test_hmt5_span_corruption_is_lossless():
    toks = "a b c d e f g h i j".split()
    r = geron_t5(toks, noise_density=0.4, mean_span=2, seed=5)
    assert restore(r["encoder_input"], r["decoder_target"]) == toks
    assert int(r["n_masked"]) == sum(l for _, l in r["spans"])
    kept = [t for t in r["encoder_input"] if not t.startswith("<extra_id_")]
    assert len(kept) + int(r["n_masked"]) == len(toks)


def test_hmxln_masks_follow_the_permutation():
    perm = [2, 0, 3, 1]
    content, query = permutation_masks(perm)
    rank = {p: i for i, p in enumerate(perm)}
    for t in range(4):
        for j in range(4):
            assert content[t, j] == float(rank[j] <= rank[t])
            assert query[t, j] == float(rank[j] < rank[t])
    r = geron_xlnet([0, 1, 2, 1], vocab_size=3, seed=2)
    assert sorted(int(v) for v in r["permutation"]) == [0, 1, 2, 3]
    assert float(r["query_mask"][int(r["permutation"][0])].sum()) == 0.0
    assert float(r["total_logprob"]) == pytest.approx(float(np.sum(r["logprobs"])), abs=1e-12)


def test_hmwpt_first_merge_maximises_the_likelihood_score():
    corpus = ["hug hug hugs pug pun bun"]
    r = geron_wordpiece_tokenizer(corpus, vocab_size=12)
    counts = Counter()
    for line in corpus:
        for w in line.split():
            counts[w] += 1
    pieces = Counter()
    pairs = Counter()
    for w, c in counts.items():
        split = [w[0]] + ["##" + ch for ch in w[1:]]
        for p in split:
            pieces[p] += c
        for i in range(len(split) - 1):
            pairs[(split[i], split[i + 1])] += c
    best = max(sorted(pairs), key=lambda k: pairs[k] / (pieces[k[0]] * pieces[k[1]]))
    assert r["merges"][0] == best
    assert "".join(t.replace("##", "") for t in r["tokenize"]("hug")) == "hug"
    assert r["tokenize"]("qqq") == ["[UNK]"]


def test_hmwemb_similarity_is_cosine():
    E = [[3.0, 4.0], [-3.0, -4.0], [4.0, -3.0]]
    r = geron_word_embeddings(["a", "b", "c"], E=E)
    assert float(r["similarity"][0, 1]) == pytest.approx(-1.0, abs=1e-12)
    assert float(r["similarity"][0, 2]) == pytest.approx(0.0, abs=1e-12)
    assert float(r["norms"][0]) == pytest.approx(5.0, abs=1e-12)
    with pytest.raises(ValueError):
        r["lookup"]("zzz")
    with pytest.raises(ValueError):
        geron_word_embeddings(["a", "a"], d=2)


# =======================================================================
# vision heads
# =======================================================================
def test_hmssg_iou_matches_brute_force_counting():
    def model(x):
        s = np.zeros((2, 3, 2))
        s[:, :2, 1] = 1.0  # predict class 1 on the left two columns
        s[:, 2:, 0] = 1.0
        return s

    truth = [[1, 1, 1], [1, 0, 0]]
    r = geron_semantic_segmentation(np.zeros((2, 3)), model, y_true=truth)
    pred = r["labels"]
    for k in (0, 1):
        inter = sum(1 for i in range(2) for j in range(3) if pred[i, j] == k and truth[i][j] == k)
        union = sum(1 for i in range(2) for j in range(3) if pred[i, j] == k or truth[i][j] == k)
        assert float(r["iou"][k]) == pytest.approx(inter / union, abs=1e-12)
    hits = sum(1 for i in range(2) for j in range(3) if pred[i, j] == truth[i][j])
    assert float(r["pixel_accuracy"]) == pytest.approx(hits / 6, abs=1e-12)


def test_hmyolo_iou_and_nms_keep_the_best_box():
    a = (0.0, 0.0, 2.0, 2.0)
    b = (1.0, 1.0, 3.0, 3.0)
    assert box_iou(a, b) == pytest.approx(1.0 / 7.0, abs=1e-12)
    assert box_iou(a, (5.0, 5.0, 6.0, 6.0)) == 0.0

    def model(x):
        p = np.zeros((1, 1, 11))  # B = 2, C = 1
        p[0, 0, :5] = [0.5, 0.5, 0.4, 0.4, 0.9]
        p[0, 0, 5:10] = [0.5, 0.5, 0.4, 0.4, 0.6]
        p[0, 0, 10] = 1.0
        return p

    r = geron_yolo(None, model, n_boxes=2)
    assert int(r["n_candidates"]) == 2
    assert int(r["n_detections"]) == 1
    assert float(r["scores"][0]) == pytest.approx(0.9, abs=1e-12)


# =======================================================================
# reinforcement learning
# =======================================================================
def test_hmrwd_return_matches_an_explicit_sum():
    # A reward table over a 2-state, 1-action chain, walked explicitly.
    table = np.zeros((2, 1, 2))
    table[0, 0, 1] = 1.0
    table[1, 0, 0] = -2.0
    table[1, 0, 1] = 3.0
    s = [0, 1, 1]
    a = [0, 0, 0]
    sp = [1, 0, 1]
    g = 0.5
    rewards = [table[s[t], a[t], sp[t]] for t in range(3)]
    want = sum(g**t * rewards[t] for t in range(3))
    r = geron_reward_function(s, a, sp, R=table, gamma=g)
    assert [float(v) for v in r["rewards"]] == pytest.approx(rewards, abs=1e-12)
    assert float(r["discounted_return"]) == pytest.approx(want, abs=1e-12)
    assert float(r["total_reward"]) == pytest.approx(sum(rewards), abs=1e-12)
    # callable form must agree with the table form
    rc = geron_reward_function(s, a, sp, R=lambda i, j, k: float(table[i, j, k]), gamma=g)
    assert float(rc["discounted_return"]) == pytest.approx(want, abs=1e-12)


def test_hmvf_agrees_with_value_iteration():
    P = np.zeros((2, 2, 2))
    P[0, 0] = [0.0, 1.0]
    P[0, 1] = [1.0, 0.0]
    P[1, 0] = [1.0, 0.0]
    P[1, 1] = [0.0, 1.0]
    R = np.zeros((2, 2, 2))
    R[0, 0, 1] = 1.0
    R[1, 0, 0] = 2.0
    pi = np.array([[0.5, 0.5], [1.0, 0.0]])
    g = 0.8
    r = geron_value_function(None, pi, g, P=P, R=R)
    # independent route: iterate the Bellman expectation operator to a fixed point
    V = np.zeros(2)
    for _ in range(2000):
        V = np.array([sum(pi[s, a] * sum(P[s, a, t] * (R[s, a, t] + g * V[t]) for t in range(2)) for a in range(2)) for s in range(2)])
    assert np.allclose(r["V"], V, atol=1e-8)
    assert float(r["residual"]) < 1e-10


def test_hmtd_update_is_hand_derivable_and_sequential():
    r = geron_td_learning([1.0, 2.0], [0, 1], [0.5, -1.0], [1, 0], alpha=0.25, gamma=0.5)
    # step 1: target = 0.5 + 0.5*2 = 1.5, error = 0.5, V0 -> 1.125
    assert float(r["target"][0]) == pytest.approx(1.5, abs=1e-12)
    assert float(r["td_error"][0]) == pytest.approx(0.5, abs=1e-12)
    # step 2 sees the *updated* V0 = 1.125: target = -1 + 0.5*1.125
    assert float(r["target"][1]) == pytest.approx(-1.0 + 0.5 * 1.125, abs=1e-12)
    assert float(r["V"][0]) == pytest.approx(1.125, abs=1e-12)
    with pytest.raises(ValueError):
        geron_td_learning([1.0], [0], [1.0], [5])


class _Bandit:
    n_states, n_actions = 1, 3

    def reset(self):
        return 0

    def step(self, a):
        return 0, float(a), False


def test_hmsac_policy_is_boltzmann_over_q():
    r = geron_sac(_Bandit(), epochs=25, lr=0.5, alpha=0.5)
    q = np.asarray(r["Q"])[0] / 0.5
    e = np.exp(q - q.max())
    assert np.allclose(r["policy"][0], e / e.sum(), atol=1e-12)
    ent = float(-np.sum(r["policy"][0] * np.log(r["policy"][0])))
    assert ent <= math.log(3) + 1e-12
    assert int(np.argmax(r["policy"][0])) == 2


def test_hmtd3_twin_target_is_never_above_the_single_critic():
    r = geron_td3(_Bandit(), epochs=20, steps=10)
    assert float(r["overestimation_gap"]) >= 0.0
    assert int(r["policy"][0]) == int(np.argmax(r["Q1"][0]))
    assert int(r["policy_updates"]) == 20 // 2


# =======================================================================
# classical supervised learning
# =======================================================================
def test_hmsgdc_first_update_and_hinge_loss():
    r = geron_sgd_classifier([[2.0, -1.0]], [1], lr=0.25, n_iter=1, alpha=0.0, shuffle=False)
    assert [float(v) for v in r["w"]] == pytest.approx([0.5, -0.25], abs=1e-12)
    assert float(r["b"]) == pytest.approx(0.25, abs=1e-12)
    X = [[3.0, 1.0], [2.0, 2.0], [-3.0, -1.0], [-2.0, -2.0]]
    y = [1, 1, 0, 0]
    r2 = geron_sgd_classifier(X, y, lr=0.05, n_iter=60)
    f = np.asarray(X) @ r2["w"] + r2["b"]
    t = np.where(np.asarray(y) == 1, 1.0, -1.0)
    hinge = float(np.mean(np.maximum(0.0, 1 - t * f)) + 0.5 * 1e-4 * float(r2["w"] @ r2["w"]))
    assert float(r2["loss_curve"][-1]) == pytest.approx(hinge, abs=1e-12)
    assert float(r2["accuracy"]) == 1.0


def test_hmsgdu_gradient_matches_finite_difference():
    X = np.array([[1.5, -2.0]])
    y = np.array([0.75])
    theta = np.array([0.3, -0.1])
    loss = lambda th: float((X[0] @ th - y[0]) ** 2)
    r = geron_sgd_update(X, y, theta, eta=0.1, index=0)
    assert np.allclose(r["gradient"], grad_fd(loss, theta), atol=1e-6)
    assert np.allclose(r["theta"], theta - 0.1 * np.asarray(r["gradient"]), atol=1e-12)


def test_hmsvdp_agrees_with_lstsq_and_is_minimum_norm():
    X = [[1.0, 2.0], [2.0, 4.1], [3.0, 5.9], [4.0, 8.2]]
    y = [1.0, 2.0, 3.1, 3.9]
    ref = np.linalg.lstsq(np.asarray(X), np.asarray(y), rcond=None)[0]
    r = geron_svd_pseudoinverse(X, y)
    assert np.allclose(r["theta"], ref, atol=1e-9)
    # rank-deficient: the returned solution must be the shortest exact one
    r2 = geron_svd_pseudoinverse([[1.0, 1.0], [2.0, 2.0]], [2.0, 4.0])
    assert int(r2["rank"]) == 1
    assert float(np.linalg.norm(r2["theta"])) <= float(np.linalg.norm([2.0, 0.0])) + 1e-12
    assert float(r2["rss"]) == pytest.approx(0.0, abs=1e-20)


def test_hmsil_matches_a_brute_force_silhouette():
    X = np.array([[0.0], [1.0], [4.0], [5.0], [10.0]])
    lab = np.array([0, 0, 1, 1, 2])
    r = geron_silhouette(X, lab)
    for i in range(5):
        own = [j for j in range(5) if lab[j] == lab[i] and j != i]
        a = sum(abs(X[i, 0] - X[j, 0]) for j in own) / len(own) if own else 0.0
        others = []
        for c in set(lab.tolist()) - {lab[i]}:
            members = [j for j in range(5) if lab[j] == c]
            others.append(sum(abs(X[i, 0] - X[j, 0]) for j in members) / len(members))
        b = min(others)
        want = 0.0 if not own else (b - a) / max(a, b)
        assert float(r["samples"][i]) == pytest.approx(want, abs=1e-12)


def test_hmsrp_entries_and_density():
    X = np.asarray(lcg(60, seed=9)).reshape(10, 6)
    r = geron_sparse_rand_projection(X, 3, density=0.5, seed=4)
    nz = r["R"][r["R"] != 0]
    assert np.allclose(np.abs(nz), math.sqrt(2.0 / 3.0), atol=1e-12)
    assert set(np.sign(nz).tolist()) <= {-1.0, 1.0}
    assert 0.2 <= float(np.count_nonzero(r["R"])) / r["R"].size <= 0.8
    assert r["X_proj"].shape == (10, 3)
    with pytest.raises(ValueError):
        geron_sparse_rand_projection(X, 99)


def test_hmstr_allocation_is_proportional_and_exact():
    strata = [0] * 10 + [1] * 6 + [2] * 4
    r = geron_stratified_sampling([[0.0]] * 20, strata, n_total=10)
    assert sum(r["allocation"].values()) == 10
    assert r["allocation"] == {0: 5, 1: 3, 2: 2}
    got = Counter(strata[i] for i in r["indices"])
    assert dict(got) == r["allocation"]
    with pytest.raises(ValueError):
        geron_stratified_sampling([[0.0]] * 20, strata, n_total=2)


def test_hmvth_matches_counter_majority():
    preds = [[1, 0, 2, 2], [1, 2, 2, 0], [0, 2, 1, 2]]
    models = [lambda X, p=p: p for p in preds]
    r = geron_voting_hard(models, [[0.0]] * 4)
    for i in range(4):
        col = [p[i] for p in preds]
        want = min(k for k, c in Counter(col).items() if c == max(Counter(col).values()))
        assert int(r["predicted"][i]) == want


def test_hmstk_meta_features_are_out_of_fold():
    X = [[float(i)] for i in range(1, 7)]
    y = [2.0 * i for i in range(1, 7)]
    seen = {"train_sizes": []}

    def spy(Xtr, ytr, Xte):
        seen["train_sizes"].append(len(Xtr))
        return np.full(len(Xte), float(np.mean(ytr)))

    def ols(Xtr, ytr, Xte):
        A = np.hstack([np.ones((len(Xtr), 1)), np.asarray(Xtr, float)])
        B = np.hstack([np.ones((len(Xte), 1)), np.asarray(Xte, float)])
        return B @ np.linalg.lstsq(A, np.asarray(ytr, float), rcond=None)[0]

    r = geron_stacking(X, y, [spy, ols], k_folds=3)
    assert all(s < 6 for s in seen["train_sizes"])  # never trained on the full set
    assert float(r["oof_mse"][1]) < float(r["oof_mse"][0])
    assert float(r["stacked_mse"]) < float(r["oof_mse"][0])


def test_hmxgb_leaf_weight_and_gain_from_the_formula():
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [1.0, 1.0, 9.0, 9.0]
    r = geron_xgboost(X, y, n_estimators=1, learning_rate=1.0, max_depth=1, reg_lambda=0.0)
    base = sum(y) / 4
    g = [base - v for v in y]
    GL, HL = g[0] + g[1], 2.0
    GR, HR = g[2] + g[3], 2.0
    assert float(r["base_score"]) == pytest.approx(base, abs=1e-12)
    assert float(r["trees"][0]["left"]["weight"]) == pytest.approx(-GL / HL, abs=1e-12)
    want_gain = 0.5 * (GL**2 / HL + GR**2 / HR - (GL + GR) ** 2 / (HL + HR))
    assert float(r["trees"][0]["gain"]) == pytest.approx(want_gain, abs=1e-12)
    assert [float(v) for v in r["predicted"]] == pytest.approx(y, abs=1e-12)


def test_hmxgb_loss_is_monotone_and_logistic_is_bounded():
    X = [[float(i)] for i in range(8)]
    y = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    r = geron_xgboost(X, y, n_estimators=6, max_depth=2, objective="logistic")
    curve = list(map(float, r["loss_curve"]))
    assert all(a >= b - 1e-12 for a, b in zip(curve, curve[1:]))
    assert float(np.min(r["predicted"])) >= 0.0 and float(np.max(r["predicted"])) <= 1.0
    with pytest.raises(ValueError):
        geron_xgboost(X, [0.0, 2.0] * 4, objective="logistic")


# =======================================================================
# clustering / unsupervised
# =======================================================================
def test_hmspcl_laplacian_and_components():
    X = [[0.0], [0.3], [8.0], [8.2]]
    r = geron_spectral_clustering(X, 2)
    L = r["laplacian"]
    assert np.allclose(L.sum(axis=1), 0.0, atol=1e-10)  # every Laplacian row sums to zero
    assert int(r["n_components"]) == 2
    lab = r["labels"]
    assert lab[0] == lab[1] and lab[2] == lab[3] and lab[0] != lab[2]


def test_hmsslc_representatives_are_real_points():
    X = [[0.0], [0.1], [0.2], [7.0], [7.1], [7.2]]
    r = geron_semisupervised_cluster(X, [[0.05], [7.05]], [3, 8], n_clusters=2, y_true=[3, 3, 3, 8, 8, 8])
    assert float(r["accuracy"]) == 1.0
    for rep in r["representatives"]:
        assert 0 <= int(rep) < 6
    assert sorted(int(v) for v in r["representative_labels"]) == [3, 8]


def test_hmuns_delegates_consistently():
    from morie.fn.hmagc import geron_agglomerative
    from morie.fn.hmsil import geron_silhouette as sil

    X = [[0.0, 0.0], [0.4, 0.4], [6.0, 6.0], [6.4, 6.4]]
    r = geron_unsupervised_learning(X, n_clusters=2, bottleneck=1)
    ref = geron_agglomerative(np.asarray(X, float), n_clusters=2, linkage="average")
    assert list(map(int, r["labels"])) == list(map(int, ref["labels"]))
    assert float(r["silhouette"]) == pytest.approx(float(sil(X, ref["labels"])["silhouette"]), abs=1e-12)
    assert float(r["recon_error"]) < 1e-20


def test_hmvbgm_digamma_and_relevance_determination():
    # psi(x+1) - psi(x) = 1/x is an identity the implementation does not use directly
    for x in (0.4, 1.0, 2.5, 7.3):
        assert float(digamma(x + 1) - digamma(x)) == pytest.approx(1.0 / x, abs=1e-9)
    assert float(digamma(1.0)) == pytest.approx(-0.5772156649015329, abs=1e-9)
    X = [[0.0], [0.1], [0.2], [8.0], [8.1], [8.2]]
    r = geron_variational_bayes_gmm(X, n_components=4, alpha0=1e-3, max_iter=300)
    assert float(np.sum(r["weights"])) == pytest.approx(1.0, abs=1e-12)
    assert int(r["n_effective"]) == 2
    assert len(set(int(v) for v in r["labels"])) == 2


def test_hmvqv_quantisation_is_nearest_neighbour():
    z = np.array([[0.0], [1.0], [4.9]])
    cb = np.array([[0.0], [5.0]])
    idx, zq = quantize(z, cb)
    for i in range(3):
        d = [abs(z[i, 0] - c[0]) for c in cb]
        assert int(idx[i]) == int(np.argmin(d))
    assert np.allclose(zq, cb[idx])
    r = geron_vq_vae([[0.0, 0.0], [0.2, 0.2], [6.0, 6.0], [6.2, 6.2]], codebook_size=2, latent_dim=1, epochs=200)
    assert np.allclose(r["z_q"], r["codebook"][r["indices"]], atol=1e-12)
    assert 1.0 <= float(r["perplexity"]) <= 2.0


# =======================================================================
# autoencoders / representation learning
# =======================================================================
def test_hmsae_backprop_matches_finite_differences():
    A = np.array([[0.1, 0.4], [0.7, 0.2], [0.3, 0.9], [0.5, 0.5]])
    u = np.asarray(lcg(6, seed=17)) - 0.5
    Ws = [u[:4].reshape(2, 2) * 0.3, u[4:6].reshape(2, 1) * 0.3]
    bs = [np.zeros(2), np.zeros(1)]
    cs = [np.zeros(2), np.zeros(2)]
    hs, rs = _forward(A, Ws, bs, cs)
    dW, db, dc = _backward(A, Ws, bs, cs, hs, rs)

    def loss(Ws, bs, cs):
        return float(np.mean((_forward(A, Ws, bs, cs)[1][0] - A) ** 2))

    for i in range(2):
        for idx in np.ndindex(Ws[i].shape):
            up = [w.copy() for w in Ws]
            dn = [w.copy() for w in Ws]
            up[i][idx] += 1e-6
            dn[i][idx] -= 1e-6
            fd = (loss(up, bs, cs) - loss(dn, bs, cs)) / 2e-6
            assert dW[i][idx] == pytest.approx(fd, abs=1e-7)


def test_hmsae_pretraining_and_finetuning_reduce_error():
    X = [[float(i), float(i)] for i in range(6)]
    r = geron_stacked_autoencoder(X, hidden_sizes=(1,), epochs=300, lr=0.2)
    assert float(r["layer_losses"][0][-1]) < float(r["layer_losses"][0][0])
    assert float(r["finetune_losses"][-1]) <= float(r["finetune_losses"][0]) + 1e-12
    assert r["codes"].shape == (6, 1)
    with pytest.raises(ValueError):
        geron_stacked_autoencoder(X, hidden_sizes=(5,))


def test_hmvae_gradients_match_finite_differences():
    X = np.array([[0.2, 0.9], [0.7, 0.1], [0.4, 0.4]])
    u = np.asarray(lcg(10, seed=23)) - 0.5
    params = [u[:2].reshape(2, 1), np.array([0.05]), u[2:4].reshape(2, 1), np.array([-0.02]), u[4:6].reshape(1, 2), np.zeros(2)]
    eps = (np.asarray(lcg(3, seed=31)) - 0.5).reshape(3, 1)
    _, _, _, grads = vae_loss_and_grads(X, params, eps, 1.0)
    for i in range(6):
        for idx in np.ndindex(params[i].shape):
            up = [p.copy() for p in params]
            dn = [p.copy() for p in params]
            up[i][idx] += 1e-6
            dn[i][idx] -= 1e-6
            fd = (vae_loss_and_grads(X, up, eps, 1.0)[0] - vae_loss_and_grads(X, dn, eps, 1.0)[0]) / 2e-6
            assert grads[i][idx] == pytest.approx(fd, abs=1e-6)


def test_hmvae_kl_is_nonnegative_and_beta_tightens_it():
    X = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    lo = geron_vae(X, latent_dim=1, epochs=200, lr=0.02, beta=0.5)
    hi = geron_vae(X, latent_dim=1, epochs=200, lr=0.005, beta=20.0)
    assert float(lo["kl"]) >= 0.0 and float(hi["kl"]) >= 0.0
    assert float(hi["kl"]) < float(lo["kl"])


def test_hmunsp_loo_matches_explicit_refits():
    Xu = [[float(i), float(i)] for i in range(6)]
    Xl = [[0.0, 0.1], [2.0, 1.9], [4.0, 4.2], [5.0, 4.8]]
    yl = [0.2, 2.1, 3.8, 5.1]
    r = geron_unsupervised_pretraining(Xu, Xl, yl, bottleneck=1)
    codes = np.asarray(r["codes"])
    D = np.hstack([np.ones((4, 1)), codes])
    errs = []
    for i in range(4):  # brute-force leave-one-out
        keep = [j for j in range(4) if j != i]
        theta = np.linalg.lstsq(D[keep], np.asarray(yl)[keep], rcond=None)[0]
        errs.append((D[i] @ theta - yl[i]) ** 2)
    assert float(r["pretrained_loo"]) == pytest.approx(float(np.mean(errs)), abs=1e-8)


def test_hmself_recovers_a_deterministic_column():
    X = [[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [3.0, 5.0, 8.0], [0.0, 4.0, 4.0], [2.0, 2.0, 4.0]]
    r = geron_self_supervised(X, "mask")
    assert float(r["task_losses"][2]) == pytest.approx(0.0, abs=1e-18)
    assert float(r["r2"][2]) == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(ValueError):
        geron_self_supervised(X, "rotate")


def test_hmsem_reduces_to_ols_and_smooths():
    Xl = [[0.0], [1.0], [2.0], [3.0]]
    yl = [1.0, 3.0, 5.0, 7.0]
    Xu = [[0.5], [1.5], [2.5]]
    r0 = geron_semisupervised(Xl, yl, Xu, alpha=0.0)
    D = np.hstack([np.ones((4, 1)), np.asarray(Xl)])
    ref = np.linalg.lstsq(D, np.asarray(yl), rcond=None)[0]
    assert np.allclose(r0["theta"], ref, atol=1e-9)
    W = r0["affinity"]
    f = np.asarray(r0["unlabeled_pred"])
    brute = 0.5 * sum(W[i, j] * (f[i] - f[j]) ** 2 for i in range(3) for j in range(3))
    assert float(r0["roughness"]) == pytest.approx(brute, abs=1e-9)
    r1 = geron_semisupervised(Xl, yl, Xu, alpha=10.0)
    assert float(r1["roughness"]) < float(r0["roughness"])


def test_hmsup_loo_matches_explicit_refits():
    X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [2.2, 3.9, 6.2, 7.8, 10.1]
    r = geron_supervised_learning(X, y)
    D = np.hstack([np.ones((5, 1)), np.asarray(X)])
    errs = []
    for i in range(5):
        keep = [j for j in range(5) if j != i]
        th = np.linalg.lstsq(D[keep], np.asarray(y)[keep], rcond=None)[0]
        errs.append((D[i] @ th - y[i]) ** 2)
    assert float(r["loo_risk"]) == pytest.approx(float(np.mean(errs)), abs=1e-9)
    assert float(r["optimism"]) > 0.0


# =======================================================================
# training diagnostics and schedules
# =======================================================================
def test_hmuf_diagnosis_table():
    assert geron_underfitting(0.5, threshold=0.1, val_err=0.51)["diagnosis"] == "underfitting"
    assert geron_underfitting(0.01, threshold=0.1, val_err=0.6)["diagnosis"] == "overfitting"
    assert geron_underfitting(0.02, threshold=0.1, val_err=0.04)["diagnosis"] == "adequate"
    assert geron_underfitting([0.9, 0.7, 0.5, 0.5, 0.5], threshold=0.1)["underfitting"] is True
    with pytest.raises(ValueError):
        geron_underfitting(0.5)


def test_hmtfl_freezes_exactly_the_leading_layers():
    W0 = np.array([[0.4, -0.3], [0.2, 0.6]])
    W1 = np.array([[0.7], [-0.2]])
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    y = [1.0, 2.0, 3.0, 4.0]
    r = geron_transfer_learning([W0.copy(), W1.copy()], X, y, n_frozen=1, epochs=200, lr=0.1)
    assert np.array_equal(r["weights"][0], W0)
    assert not np.array_equal(r["weights"][1], W1)
    assert int(r["trainable_params"]) == W1.size
    assert int(r["total_params"]) == W0.size + W1.size
    assert float(r["final_loss"]) < float(r["initial_loss"])


def test_hmvgr_and_hmxgr_ratios_by_hand():
    grads = [[1e-4], [1e-2], [1.0]]
    r = geron_vanishing_gradients(grads)
    assert float(r["geometric_ratio"]) == pytest.approx(0.01, abs=1e-9)
    assert float(r["attenuation"]) == pytest.approx(1e-4, abs=1e-12)
    assert r["vanishing"] is True
    x = geron_exploding_gradients([[100.0], [10.0], [1.0]], tol=2.0)
    assert float(x["geometric_ratio"]) == pytest.approx(10.0, abs=1e-9)
    assert x["exploding"] is True
    clip = geron_exploding_gradients([[3.0, 0.0], [0.0, 4.0]], clip_norm=2.5)
    assert float(clip["global_norm"]) == pytest.approx(5.0, abs=1e-12)
    total = math.sqrt(sum(float(np.sum(np.asarray(g) ** 2)) for g in clip["clipped"]))
    assert total == pytest.approx(2.5, abs=1e-12)


def test_hmwrst_cosine_values_at_known_points():
    steps = [0, 5, 9, 10, 20, 29]
    r = geron_warm_restarts(steps, T0=10, factor=2.0, eta_max=0.2, eta_min=0.02)
    lengths = [10, 10, 10, 20, 20, 20]
    starts = [0, 0, 0, 10, 10, 10]
    for k, step in enumerate(steps):
        cur = step - starts[k]
        want = 0.02 + 0.5 * (0.2 - 0.02) * (1 + math.cos(math.pi * cur / lengths[k]))
        assert float(r["eta"][k]) == pytest.approx(want, abs=1e-12)
    assert [int(c) for c in r["cycle"]] == [0, 0, 0, 1, 1, 1]


# =======================================================================
# fine-tuning
# =======================================================================
def test_hmsft_gradient_and_convergence():
    data = [("alpha beta", "yes"), ("gamma delta", "no"), ("alpha delta", "yes")]
    r = geron_sft(None, data, epochs=1, lr=1e-12)
    assert float(r["loss_curve"][0]) == pytest.approx(math.log(2), abs=1e-12)
    r2 = geron_sft(None, data, epochs=400, lr=0.5)
    assert float(r2["loss"]) < float(r["loss_curve"][0])
    assert float(r2["accuracy"]) == 1.0
    assert float(r2["sum_loss"]) == pytest.approx(float(r2["loss"]) * 3, abs=1e-12)


def test_hmtrlf_dpo_loss_matches_hand_log_sigmoid():
    pairs = [([1.0, 0.0], [0.0, 1.0])]
    r = geron_trl_finetune(None, pairs, method="dpo", epochs=1, lr=1e-12, beta=2.0)
    assert float(r["loss_curve"][0]) == pytest.approx(-math.log(0.5), abs=1e-12)
    r2 = geron_trl_finetune(None, pairs, method="dpo", epochs=200, lr=0.5, beta=2.0)
    d = np.array([1.0, -1.0])
    want = -math.log(1 / (1 + math.exp(-2.0 * float(d @ r2["theta"]))))
    assert float(r2["loss"]) == pytest.approx(want, abs=1e-9)
    assert float(r2["margin"]) > 0.0


def test_hmtrlf_ppo_clip_stops_the_update():
    items = [([1.0], 0.0, 2.0)]
    r = geron_trl_finetune(None, items, method="ppo", epochs=100, lr=0.5, clip_eps=0.1)
    assert float(r["ratio"][0]) >= 1.1
    assert float(r["loss"]) == pytest.approx(-1.1 * 2.0, abs=1e-9)
    assert float(r["clipped_fraction"]) == 1.0
    # a negative advantage moves the other way and is clipped at 1 - eps
    r2 = geron_trl_finetune(None, [([1.0], 0.0, -2.0)], method="ppo", epochs=100, lr=0.5, clip_eps=0.1)
    assert float(r2["ratio"][0]) <= 0.9 + 1e-9


# =======================================================================
# systems / tooling
# =======================================================================
def test_hmtsc_replay_equals_manual_ops():
    W = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([0.5, -0.5])
    x = np.array([[1.0, -1.0]])
    r = geron_torchscript([("linear", W), ("bias", b), ("relu",)], x)
    want = np.maximum(x @ W + b, 0.0)
    assert np.allclose(r["output"], want, atol=1e-12)
    assert np.allclose(r["replay"], want, atol=1e-12)
    with pytest.raises(ValueError):
        run_graph(r["graph"], np.zeros((1, 5)))


def test_hmtcmp_fusion_is_exact_and_dp_matches_brute_force():
    A = np.array([[1.0, 2.0], [0.0, 1.0]])
    B = np.array([[0.0, 1.0], [1.0, 0.0]])
    C = np.array([[2.0, 0.0], [0.0, 3.0]])
    x = np.array([[1.0, -1.0]])
    r = geron_torch_compile([("linear", A), ("linear", B), ("linear", C)], example_inputs=x)
    assert int(r["n_compiled"]) == 1
    assert np.allclose(r["compiled"][0]["op"][1], A @ B @ C, atol=1e-12)
    assert np.allclose(r["output"], x @ A @ B @ C, atol=1e-12)

    dims = [7, 3, 11, 2, 5]  # brute force every parenthesisation
    def brute(i, j):
        if j == i + 1:
            return 0
        return min(brute(i, k) + brute(k, j) + dims[i] * dims[k] * dims[j] for k in range(i + 1, j))

    assert matmul_order(dims)[0] == pytest.approx(float(brute(0, len(dims) - 1)), abs=1e-9)


def test_hmtpp_sharding_is_exact_in_both_schemes():
    W = np.asarray(lcg(24, seed=41)).reshape(4, 6) * 2 - 1
    x = np.asarray(lcg(8, seed=43)).reshape(2, 4)
    ref = x @ W
    col = geron_tensor_parallelism(W, 3, x=x, scheme="column")
    row = geron_tensor_parallelism(W, 2, x=x, scheme="row")
    assert np.allclose(col["output"], ref, atol=1e-12)
    assert np.allclose(row["output"], ref, atol=1e-12)
    assert int(col["comm_elements"]) == 0
    assert int(row["comm_elements"]) == 2 * ref.size
    with pytest.raises(ValueError):
        geron_tensor_parallelism(W, 5, x=x)


def test_hmsvm2_round_trip_and_key_checking():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "sd.npz")
    sd = {"a": np.arange(6, dtype=float).reshape(2, 3), "b": np.array([1, 2, 3], dtype=np.int64)}
    r = geron_save_load_pytorch(sd, p)
    assert r["exact"] is True
    assert int(r["n_params"]) == 9
    assert np.array_equal(r["loaded"]["b"], sd["b"])
    assert r["dtypes"]["b"] == "int64"
    with pytest.raises(ValueError):
        geron_save_load_pytorch({}, p)
    with pytest.raises(ValueError):
        geron_save_load_pytorch(sd, os.path.join(d, "no_such_dir", "x.npz"))


# =======================================================================
# manifold learning
# =======================================================================
def test_hmtsne_rows_hit_the_requested_perplexity():
    X = [[0.0], [1.0], [2.5], [6.0], [6.4], [9.0]]
    perp = 3.0
    r = geron_tsne(X, n_components=1, perplexity=perp, n_iter=50)
    # independent check of the sigma binary search: recompute row entropies
    A = np.asarray(X, float)
    D2 = np.sum((A[:, None, :] - A[None, :, :]) ** 2, axis=2)
    for i in range(6):
        idx = [j for j in range(6) if j != i]
        w = np.exp(-D2[i, idx] * r["betas"][i])
        p = w / w.sum()
        H = float(-np.sum(p * np.log(p)))
        assert H == pytest.approx(math.log(perp), abs=1e-4)
    assert float(r["P"].sum()) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(r["P"], r["P"].T, atol=1e-15)


def test_hmumap_rho_and_membership_normalisation():
    X = [[0.0], [1.0], [3.0], [7.0], [7.5], [12.0]]
    k = 3
    r = geron_umap(X, n_components=1, n_neighbors=k, n_iter=50)
    A = np.asarray(X, float)
    D = np.abs(A - A.T)
    for i in range(6):
        nn = sorted(D[i][D[i] > 0]) if np.any(D[i] > 0) else [0.0]
        assert float(r["rho"][i]) == pytest.approx(nn[0], abs=1e-12)
        order = np.argsort(D[i], kind="mergesort")[1 : k + 1]
        tot = float(np.sum(np.exp(-np.maximum(D[i, order] - r["rho"][i], 0.0) / r["sigma"][i])))
        assert tot == pytest.approx(math.log2(k), abs=1e-4)
    assert float(r["a"]) > 0 and float(r["b"]) > 0
    assert np.all(r["graph"] >= 0) and np.all(r["graph"] <= 1)


# =======================================================================
# symbolic differentiation
# =======================================================================
@pytest.mark.parametrize(
    "expr,point",
    [
        ("x^3 + 2*x", 1.3),
        ("sin(x)*cos(x)", 0.7),
        ("exp(2*x) / (1 + x)", 0.4),
        ("log(x^2 + 1)", 2.1),
        ("tanh(3*x)", -0.6),
        ("sqrt(x + 2)", 1.0),
    ],
)
def test_hmsymd_matches_central_differences(expr, point):
    r = geron_symbolic_diff(expr, "x", at={"x": point})
    tree = parse(expr)
    h = 1e-6
    fd = (evaluate(tree, {"x": point + h}) - evaluate(tree, {"x": point - h})) / (2 * h)
    assert float(r["value"]) == pytest.approx(float(fd), abs=1e-5)
    assert float(r["error"]) < 1e-5


def test_hmsymd_rejects_nonsense():
    with pytest.raises(ValueError):
        geron_symbolic_diff("x +", "x")
    with pytest.raises(ValueError):
        geron_symbolic_diff("frobnicate(x)", "x")
    assert geron_symbolic_diff("7", "x")["derivative"] == "0"


# =======================================================================
# multimodal
# =======================================================================
def test_hmvbrt_attention_and_cross_modal_mass():
    r = geron_videobert([0, 1, 2], [0, 1, 1], d_model=4, mask_positions=[1, 4])
    A = np.asarray(r["attention"])
    assert A.shape == (6, 6)
    assert np.allclose(A.sum(axis=1), 1.0, atol=1e-12)
    modality = np.array([0, 0, 0, 1, 1, 1])
    brute = float(np.mean([A[i, modality != modality[i]].sum() for i in range(6)]))
    assert float(r["cross_modal_mass"]) == pytest.approx(brute, abs=1e-12)
    assert list(r["masked"]) == [1, 4]
    assert float(r["loss"]) == pytest.approx(float(np.mean(r["token_losses"])), abs=1e-12)


def test_hmvilb_costreams_have_swapped_queries():
    img = [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]]
    txt = [[0.5, -0.5], [1.0, 1.0]]
    r = geron_vilbert(img, txt, d_model=2)
    assert r["attention_v2t"].shape == (3, 2)
    assert r["attention_t2v"].shape == (2, 3)
    assert np.allclose(r["attention_v2t"].sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(r["attention_t2v"].sum(axis=1), 1.0, atol=1e-12)
    assert r["image_out"].shape == (3, 2) and r["text_out"].shape == (2, 2)


def test_hmsent_metrics_match_brute_force_counting():
    lex = {"good": 1, "bad": -1}

    def model(toks):
        s = sum(lex.get(t, 0) for t in toks)
        return [-s, s]

    texts = ["good good", "bad", "neutral words", "good bad"]
    gold = [1, 0, 1, 0]
    r = geron_sentiment_analysis(texts, model, y_true=gold)
    pred = [int(v) for v in r["predicted"]]
    acc = sum(1 for a, b in zip(pred, gold) if a == b) / 4
    assert float(r["accuracy"]) == pytest.approx(acc, abs=1e-12)
    f1s = []
    for k in (0, 1):
        tp = sum(1 for a, b in zip(pred, gold) if a == k and b == k)
        fp = sum(1 for a, b in zip(pred, gold) if a == k and b != k)
        fn = sum(1 for a, b in zip(pred, gold) if a != k and b == k)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    assert float(r["macro_f1"]) == pytest.approx(sum(f1s) / 2, abs=1e-12)


def test_hmzsl_calibration_removes_a_constant_bias():
    raw = {"a": 2.0, "b": 0.0}
    r = geron_zero_shot(lambda p: raw, "prompt")
    denom = math.exp(2.0) + 1.0
    assert [float(v) for v in r["probabilities"]] == pytest.approx([math.exp(2.0) / denom, 1 / denom], abs=1e-12)
    assert r["predicted_label"] == "a"
    f = lambda p: ([1.0, 4.0] if p == "" else [3.0, 6.0])
    rc = geron_zero_shot(f, "x", labels=["u", "v"], null_prompt="")
    assert [float(v) for v in rc["probabilities"]] == pytest.approx([0.5, 0.5], abs=1e-12)
    with pytest.raises(ValueError):
        geron_zero_shot(lambda p: [1.0], "x", labels=["only"])


# =======================================================================
# the stub these modules replaced must not pass
# =======================================================================
MEAN_OF_INPUTS_CASES = [
    (lambda: geron_sigmoid([1.0, 2.0, 3.0]), "a", 2.0),
    (lambda: geron_tanh([1.0, 2.0, 3.0]), "a", 2.0),
    (lambda: geron_selu([1.0, 2.0, 3.0]), "a", 2.0),
    (lambda: geron_swish([1.0, 2.0, 3.0]), "a", 2.0),
    (lambda: geron_softmax_function([1.0, 2.0, 3.0]), "p", 2.0),
    (lambda: geron_standardization([1.0, 2.0, 3.0]), "X_std", 2.0),
]


@pytest.mark.parametrize("call,key,input_mean", MEAN_OF_INPUTS_CASES)
def test_placeholder_shape_is_gone(call, key, input_mean):
    """The old stubs returned the mean of the inputs and an `se`; these do not."""
    r = call()
    assert key in r, f"real payload key {key!r} missing"
    assert "se" not in r, "the placeholder's standard-error key is still present"
    assert float(r["estimate"]) != pytest.approx(input_mean, abs=1e-9)
    assert "method" in r and "n" in r


def test_every_module_exposes_a_cheatsheet():
    import importlib

    names = [
        "hmrvn", "hmrwd", "hmsac", "hmsae", "hmsatt", "hmsdp", "hmself", "hmselu", "hmsem", "hmsenet",
        "hmsent", "hmseq2", "hmsft", "hmsftm", "hmsfts", "hmsgdc", "hmsgdu", "hmsigm", "hmsil", "hmspcl",
        "hmsrnn", "hmsrp", "hmssg", "hmsslc", "hmstk", "hmstr", "hmstr2", "hmstz", "hmsup", "hmsvdp",
        "hmsvm2", "hmswi", "hmswin", "hmsymd", "hmt5", "hmtanh", "hmtcmp", "hmtd", "hmtd3", "hmtfl",
        "hmtfm", "hmtlu", "hmtpp", "hmtrlf", "hmtsc", "hmtsf", "hmtsne", "hmuf", "hmumap", "hmuns",
        "hmunsp", "hmvae", "hmvbgm", "hmvbrt", "hmvf", "hmvgr", "hmvilb", "hmvit", "hmvqv", "hmvth",
        "hmwemb", "hmwpt", "hmwrst", "hmxav", "hmxcpt", "hmxgb", "hmxgr", "hmxln", "hmyolo", "hmzsl",
    ]
    assert len(names) == 70
    for name in names:
        mod = importlib.import_module("morie.fn." + name)
        text = mod.cheatsheet()
        assert text.startswith(name + ": ") and len(text) > len(name) + 4
