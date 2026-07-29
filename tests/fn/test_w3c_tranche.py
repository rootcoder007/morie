# morie.fn -- test file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent checks for the Kamath (2024) w3c tranche.

Every test recomputes the quantity by a DIFFERENT route from the
module under test -- a hand loop, a brute-force enumeration, a
mathematical invariant, or a sibling module -- so a stub that returns
the mean of its inputs cannot pass.
"""

import math
from collections import Counter
from itertools import combinations, permutations

import numpy as np
import pytest


def lcg(seed=1):
    """The package's deterministic generator: no RNG state, same
    numbers everywhere."""
    s = seed

    def nxt():
        nonlocal s
        s = (1664525 * s + 1013904223) % 2 ** 32
        return (s + 0.5) / 2 ** 32
    return nxt


def rand_matrix(rows, cols, seed=1, lo=-1.0, hi=1.0):
    r = lcg(seed)
    return np.array([[lo + (hi - lo) * r() for _ in range(cols)]
                     for _ in range(rows)])


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


# ── kmfait ────────────────────────────────────────────────────────
def test_kmfait_counts_supported_claims():
    from morie.fn.kmfait import kamath_ragas_faithfulness
    claims = ["alpha beta", "gamma", "beta gamma", "delta"]
    ctx = "alpha beta gamma"
    out = kamath_ragas_faithfulness(claims, ctx)
    # Independent: a claim is supported iff every token is in the ctx set.
    ctx_set = set(ctx.split())
    want = sum(all(t in ctx_set for t in c.split()) for c in claims)
    assert out["supported"] == want == 3
    assert out["estimate"] == pytest.approx(3 / 4)


def test_kmfait_custom_judge_and_empty_refused():
    from morie.fn.kmfait import kamath_ragas_faithfulness
    out = kamath_ragas_faithfulness(["x", "y"], "ctx",
                                    entails=lambda c, k: c == "x")
    assert out["estimate"] == 0.5
    with pytest.raises(ValueError):
        kamath_ragas_faithfulness([], "ctx")


# ── kmfew ─────────────────────────────────────────────────────────
def test_kmfew_matches_bruteforce_cosine_ranking():
    from morie.fn.kmfew import kamath_few_shot_exemplar_selection
    D = rand_matrix(9, 4, seed=3)
    q = rand_matrix(1, 4, seed=11)[0]
    out = kamath_few_shot_exemplar_selection(D, q, 3)
    sims = [float(np.dot(d, q) / (np.linalg.norm(d) * np.linalg.norm(q)))
            for d in D]
    want = sorted(range(len(D)), key=lambda i: (-sims[i], i))[:3]
    assert out["selected"] == want
    assert out["similarities"] == pytest.approx([sims[i] for i in want])


def test_kmfew_rejects_bad_k():
    from morie.fn.kmfew import kamath_few_shot_exemplar_selection
    with pytest.raises(ValueError):
        kamath_few_shot_exemplar_selection([[1.0, 0.0]], [1.0, 0.0], 5)


# ── kmfst ─────────────────────────────────────────────────────────
def test_kmfst_sums_every_ngram_occurrence():
    from morie.fn.kmfst import kamath_fasttext_subword, word_ngrams
    grams = word_ngrams("cat", 3, 3)
    tbl = {g: [1.0, float(i)] for i, g in enumerate(grams)}
    out = kamath_fasttext_subword("cat", tbl, 3, 3)
    want = np.sum([tbl[g] for g in grams], axis=0)
    assert out["vector"] == pytest.approx(list(want))
    assert out["n_known"] == len(grams)
    assert out["n_missing"] == 0


def test_kmfst_reports_missing_and_refuses_total_oov():
    from morie.fn.kmfst import kamath_fasttext_subword
    out = kamath_fasttext_subword("ab", {"<a": [1.0]}, 2, 2)
    assert out["n_known"] == 1 and out["n_missing"] >= 1
    assert out["vector"] == [1.0]
    with pytest.raises(ValueError):
        kamath_fasttext_subword("zz", {"<a": [1.0]}, 2, 2)


# ── kmgev ─────────────────────────────────────────────────────────
def test_kmgev_is_the_probability_weighted_mean():
    from morie.fn.kmgev import kamath_g_eval
    logits = [0.5, -1.0, 2.0]
    scores = [1, 3, 5]
    out = kamath_g_eval("x", "y", scores, lambda a, b, c: logits)
    e = [math.exp(v) for v in logits]
    Z = sum(e)
    want = sum(s * (v / Z) for s, v in zip(scores, e))
    assert out["estimate"] == pytest.approx(want)
    # Not the mean of the score points, and not the mean of the logits.
    assert out["estimate"] != pytest.approx(np.mean(scores))


def test_kmgev_rejects_wrong_shape():
    from morie.fn.kmgev import kamath_g_eval
    with pytest.raises(ValueError):
        kamath_g_eval("x", "y", [1, 2, 3], lambda a, b, c: [0.0, 0.0])


# ── kmglv ─────────────────────────────────────────────────────────
def test_kmglv_matches_an_explicit_double_loop():
    from morie.fn.kmglv import kamath_glove_cost
    X = np.array([[3.0, 0.0], [1.0, 200.0]])
    W = rand_matrix(2, 2, seed=5)
    Wt = rand_matrix(2, 2, seed=9)
    b = np.array([0.3, -0.7])
    bt = np.array([1.1, 0.2])
    x_max, alpha = 100.0, 0.75
    want = 0.0
    for i in range(2):
        for j in range(2):
            if X[i, j] == 0:
                continue
            f = (X[i, j] / x_max) ** alpha if X[i, j] < x_max else 1.0
            r = np.dot(W[i], Wt[j]) + b[i] + bt[j] - math.log(X[i, j])
            want += f * r * r
    out = kamath_glove_cost(X, W, Wt, b, bt, x_max, alpha)
    assert out["estimate"] == pytest.approx(want)
    assert out["n_nonzero"] == 3


def test_kmglv_weight_saturates_at_x_max():
    from morie.fn.kmglv import glove_weight
    assert float(glove_weight(np.array(200.0), 100.0, 0.75)) == 1.0
    assert float(glove_weight(np.array(50.0), 100.0, 0.75)) == pytest.approx(
        0.5 ** 0.75)


# ── kmgrnd ────────────────────────────────────────────────────────
def test_kmgrnd_counts_occurrences_not_types():
    from morie.fn.kmgrnd import kamath_groundedness_reward
    y = ["a", "a", "b", "z"]
    ctx = ["a", "b", "c"]
    out = kamath_groundedness_reward(y, ctx)
    want = sum(1 for t in y if t in set(ctx)) / len(y)
    assert out["estimate"] == pytest.approx(want) == pytest.approx(0.75)
    assert out["ungrounded"] == ["z"]


def test_kmgrnd_refuses_empty():
    from morie.fn.kmgrnd import kamath_groundedness_reward
    with pytest.raises(ValueError):
        kamath_groundedness_reward([], ["a"])


# ── kmhyb ─────────────────────────────────────────────────────────
def test_kmhyb_endpoints_recover_each_arm():
    from morie.fn.kmhyb import kamath_hybrid_retrieval_fusion
    d = [0.1, 0.9, 0.4]
    s = [2.0, 0.0, 1.0]
    assert kamath_hybrid_retrieval_fusion(d, s, 1.0)["scores"] == \
        pytest.approx(d)
    assert kamath_hybrid_retrieval_fusion(d, s, 0.0)["scores"] == \
        pytest.approx(s)
    mid = kamath_hybrid_retrieval_fusion(d, s, 0.25)
    want = [0.25 * a + 0.75 * b for a, b in zip(d, s)]
    assert mid["scores"] == pytest.approx(want)
    assert mid["ranking"] == sorted(range(3), key=lambda i: -want[i])


def test_kmhyb_rejects_lambda_outside_unit_interval():
    from morie.fn.kmhyb import kamath_hybrid_retrieval_fusion
    with pytest.raises(ValueError):
        kamath_hybrid_retrieval_fusion([1.0], [1.0], 1.5)


# ── kmhyde ────────────────────────────────────────────────────────
def test_kmhyde_ranks_by_the_hypothetical_not_the_query():
    from morie.fn.kmhyde import kamath_hyde_hypothetical_doc
    docs = [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]
    out = kamath_hyde_hypothetical_doc(
        "q", lambda q: "hypo", docs, embed=lambda t: [0.0, 1.0], k=3)
    v = np.array([0.0, 1.0])
    sims = [float(np.dot(d, v) / np.linalg.norm(d)) for d in docs]
    assert out["retrieved"] == sorted(range(3), key=lambda i: (-sims[i], i))
    assert out["retrieved"][0] == 1


def test_kmhyde_dict_corpus_keeps_ids():
    from morie.fn.kmhyde import kamath_hyde_hypothetical_doc
    out = kamath_hyde_hypothetical_doc(
        "q", lambda q: [1.0, 0.0], {"x": [1.0, 0.0], "y": [0.0, 1.0]}, k=1)
    assert out["retrieved"] == ["x"]


# ── kmicl ─────────────────────────────────────────────────────────
def test_kmicl_prompt_contains_every_demonstration_in_order():
    from morie.fn.kmicl import kamath_in_context_learning_prob
    demos = ["a -> 1", "b -> 2", "c -> 3"]
    seen = {}

    def model(prompt, y):
        seen["p"] = prompt
        return 0.125

    out = kamath_in_context_learning_prob(demos, "d ->", model, answer="4")
    assert seen["p"] == "a -> 1\nb -> 2\nc -> 3\nd ->"
    assert out["K"] == 3
    assert out["log_prob"] == pytest.approx(math.log(0.125))


def test_kmicl_rejects_non_probability():
    from morie.fn.kmicl import kamath_in_context_learning_prob
    with pytest.raises(ValueError):
        kamath_in_context_learning_prob(["a"], "b", lambda p, y: 3.4)


# ── kminst ────────────────────────────────────────────────────────
def test_kminst_scores_only_the_response_positions():
    from morie.fn.kminst import kamath_instruction_tuning_loss
    logits = rand_matrix(5, 4, seed=13)
    targets = [0, 1, 2, 3, 0]
    mask = [0, 0, 1, 1, 1]
    out = kamath_instruction_tuning_loss(logits, mask, targets)
    want = np.mean([-math.log(softmax(logits[t])[targets[t]])
                    for t in range(5) if mask[t]])
    assert out["estimate"] == pytest.approx(want)
    assert out["n_response_tokens"] == 3
    # Masking must matter: the all-positions loss differs.
    allpos = kamath_instruction_tuning_loss(logits, [1] * 5, targets)
    assert allpos["estimate"] != pytest.approx(out["estimate"])


def test_kminst_empty_mask_refused():
    from morie.fn.kminst import kamath_instruction_tuning_loss
    with pytest.raises(ValueError):
        kamath_instruction_tuning_loss([[0.0, 0.0]], [0], [1])


# ── kmitc ─────────────────────────────────────────────────────────
def test_kmitc_matches_a_hand_written_infonce():
    from morie.fn.kmitc import kamath_image_text_contrastive
    I = rand_matrix(3, 3, seed=21)
    T = rand_matrix(3, 3, seed=29)
    tau = 0.4
    In = I / np.linalg.norm(I, axis=1, keepdims=True)
    Tn = T / np.linalg.norm(T, axis=1, keepdims=True)
    S = In @ Tn.T / tau
    rows = np.mean([-math.log(softmax(S[i])[i]) for i in range(3)])
    cols = np.mean([-math.log(softmax(S[:, j])[j]) for j in range(3)])
    out = kamath_image_text_contrastive(I, T, tau)
    assert out["estimate"] == pytest.approx(0.5 * (rows + cols))


def test_kmitc_single_pair_refused():
    from morie.fn.kmitc import kamath_image_text_contrastive
    with pytest.raises(ValueError):
        kamath_image_text_contrastive([[1.0, 0.0]], [[1.0, 0.0]], 1.0)


# ── kmitm ─────────────────────────────────────────────────────────
def test_kmitm_is_a_logistic_head_on_the_concatenation():
    from morie.fn.kmitm import kamath_image_text_matching
    I, T = [0.5, -1.0], [2.0, 0.25]
    W, b = [1.0, 2.0, -0.5, 4.0], 0.75
    z = sum(w * v for w, v in zip(W, I + T)) + b
    out = kamath_image_text_matching(I, T, W, b)
    assert out["logit"] == pytest.approx(z)
    assert out["estimate"] == pytest.approx(1 / (1 + math.exp(-z)))
    assert out["match"] is (z >= 0)


def test_kmitm_width_mismatch_refused():
    from morie.fn.kmitm import kamath_image_text_matching
    with pytest.raises(ValueError):
        kamath_image_text_matching([1.0], [1.0], [1.0], 0.0)


# ── kmklr ─────────────────────────────────────────────────────────
def test_kmklr_subtracts_beta_times_kl_elementwise():
    from morie.fn.kmklr import kamath_kl_reward_shaping
    r, kl, beta = [1.0, -2.0, 0.5], [0.25, 1.0, 0.0], 0.8
    out = kamath_kl_reward_shaping(r, kl, beta)
    want = [a - beta * b for a, b in zip(r, kl)]
    assert out["shaped"] == pytest.approx(want)
    assert out["estimate"] == pytest.approx(np.mean(want))
    assert out["estimate"] != pytest.approx(np.mean(r))


def test_kmklr_negative_kl_refused():
    from morie.fn.kmklr import kamath_kl_reward_shaping
    with pytest.raises(ValueError):
        kamath_kl_reward_shaping([1.0], [-0.1], 1.0)


# ── kmlb ──────────────────────────────────────────────────────────
def test_kmlb_is_minimised_at_perfect_balance():
    from morie.fn.kmlb import kamath_moe_load_balance_loss
    N, alpha = 4, 0.01
    bal = kamath_moe_load_balance_loss([0.25] * 4, [0.25] * 4, N, alpha)
    skew = kamath_moe_load_balance_loss([0.7, 0.1, 0.1, 0.1],
                                        [0.6, 0.2, 0.1, 0.1], N, alpha)
    assert bal["estimate"] == pytest.approx(alpha)
    want = alpha * N * sum(f * p for f, p in
                           zip([0.7, 0.1, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1]))
    assert skew["estimate"] == pytest.approx(want)
    assert skew["estimate"] > bal["estimate"]


def test_kmlb_requires_distributions():
    from morie.fn.kmlb import kamath_moe_load_balance_loss
    with pytest.raises(ValueError):
        kamath_moe_load_balance_loss([0.5, 0.2], [0.5, 0.5], 2, 0.01)


# ── kmlora ────────────────────────────────────────────────────────
def test_kmlora_equals_the_merged_weight_matrix():
    from morie.fn.kmlora import kamath_lora_weight_update
    d, k, r = 3, 4, 2
    W0 = rand_matrix(d, k, seed=31)
    A = rand_matrix(r, k, seed=37)
    B = rand_matrix(d, r, seed=41)
    x = rand_matrix(1, k, seed=43)[0]
    alpha = 8.0
    merged = W0 + (alpha / r) * (B @ A)
    out = kamath_lora_weight_update(W0, A, B, alpha, r, x)
    assert out["h"] == pytest.approx(list(merged @ x))
    assert out["n_trainable"] == r * k + d * r


def test_kmlora_wrong_rank_refused():
    from morie.fn.kmlora import kamath_lora_weight_update
    with pytest.raises(ValueError):
        kamath_lora_weight_update([[1.0]], [[1.0]], [[1.0]], 1.0, 2, [1.0])


# ── kmlv ──────────────────────────────────────────────────────────
def test_kmlv_projects_and_prepends_visual_tokens():
    from morie.fn.kmlv import kamath_llava_visual_instruction
    feats = rand_matrix(3, 2, seed=47)
    W = rand_matrix(4, 2, seed=53)
    txt = rand_matrix(2, 4, seed=59)
    out = kamath_llava_visual_instruction("im", W, lambda i: feats, txt)
    assert np.allclose(out["visual_tokens"], feats @ W.T)
    assert np.allclose(out["inputs"][:3], feats @ W.T)
    assert np.allclose(out["inputs"][3:], txt)
    assert (out["n_visual"], out["n_text"]) == (3, 2)


def test_kmlv_loss_needs_both_head_and_targets():
    from morie.fn.kmlv import kamath_llava_visual_instruction
    with pytest.raises(ValueError):
        kamath_llava_visual_instruction(
            "im", [[1.0]], lambda i: [[1.0]], [[1.0]],
            lm_head=lambda z: [[0.0, 0.0], [0.0, 0.0]])


# ── kmmae ─────────────────────────────────────────────────────────
def test_kmmae_sums_squared_error_across_modalities():
    from morie.fn.kmmae import kamath_multimodal_mae
    out = kamath_multimodal_mae(
        {"img": [0.0], "txt": [0.0]},
        {"img": [1.0, 2.0], "txt": [5.0]},
        {"img": [True, True, False], "txt": [False, True]},
        decoders={"img": lambda v, m: [0.0, 0.0],
                  "txt": lambda v, m: [4.0]})
    assert out["per_modality"]["img"] == pytest.approx(1.0 + 4.0)
    assert out["per_modality"]["txt"] == pytest.approx(1.0)
    assert out["estimate"] == pytest.approx(6.0)
    assert out["n_masked"] == 3


def test_kmmae_requires_decoders_and_a_mask():
    from morie.fn.kmmae import kamath_multimodal_mae
    with pytest.raises(ValueError):
        kamath_multimodal_mae([0.0], [1.0], [True])
    with pytest.raises(ValueError):
        kamath_multimodal_mae([0.0], [1.0], [False],
                              decoders=lambda v, m: [0.0])


# ── kmmamb ────────────────────────────────────────────────────────
def test_kmmamb_matches_a_hand_rolled_scan():
    from morie.fn.kmmamb import kamath_mamba_ssm
    x = [0.5, -1.0, 2.0, 0.25]
    A = [-0.5, -2.0]
    B = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 1.0]]
    C = [[1.0, 1.0]] * 4
    delta = [0.1, 0.2, 0.3, 0.4]
    h = np.zeros(2)
    want = []
    for t in range(4):
        h = np.exp(delta[t] * np.array(A)) * h + delta[t] * np.array(B[t]) * x[t]
        want.append(float(np.dot(C[t], h)))
    out = kamath_mamba_ssm(x, A, B, C, delta)
    assert out["y"] == pytest.approx(want)


def test_kmmamb_refuses_dense_A_and_bad_delta():
    from morie.fn.kmmamb import kamath_mamba_ssm
    with pytest.raises(ValueError):
        kamath_mamba_ssm([1.0], [[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0],
                         [1.0, 1.0], 1.0)
    with pytest.raises(ValueError):
        kamath_mamba_ssm([1.0], [0.0], [1.0], [1.0], 0.0)


# ── kmmbi ─────────────────────────────────────────────────────────
def test_kmmbi_threshold_and_rates():
    from morie.fn.kmmbi import kamath_membership_inference
    losses = [0.1, 0.4, 0.6, 2.0, 0.5]
    labels = [1, 1, 0, 0, 1]
    tau = 0.5
    out = kamath_membership_inference(losses, tau, labels=labels)
    pred = [1 if L < tau else 0 for L in losses]
    assert out["predictions"] == pred
    tp = sum(p == 1 and y == 1 for p, y in zip(pred, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(pred, labels))
    assert out["tpr"] == pytest.approx(tp / 3)
    assert out["fpr"] == pytest.approx(fp / 2)
    # loss exactly at tau is a NON-member (strict inequality)
    assert pred[4] == 0


def test_kmmbi_single_class_labels_refused():
    from morie.fn.kmmbi import kamath_membership_inference
    with pytest.raises(ValueError):
        kamath_membership_inference([0.1, 0.2], 1.0, labels=[1, 1])


# ── kmmedu ────────────────────────────────────────────────────────
def test_kmmedu_predicts_each_head_argmax():
    from morie.fn.kmmedu import kamath_medusa_heads
    h = np.array([0.5, -1.5])
    W = [rand_matrix(2, 3, seed=61), rand_matrix(2, 3, seed=67),
         rand_matrix(2, 3, seed=71)]
    out = kamath_medusa_heads(h, W, 3)
    want = [int(np.argmax(h @ w)) for w in W]
    assert out["tokens"] == want
    for i, w in enumerate(W):
        assert out["probabilities"][i] == pytest.approx(
            float(softmax(h @ w).max()))


def test_kmmedu_verification_stops_at_first_reject():
    from morie.fn.kmmedu import kamath_medusa_heads
    W = [rand_matrix(2, 3, seed=61)] * 4
    out = kamath_medusa_heads([0.5, -1.5], W, 4,
                              verify=lambda i, t: i != 1)
    assert out["accepted"] == 1
    with pytest.raises(ValueError):
        kamath_medusa_heads([0.5, -1.5], W, 9)


# ── kmmoe ─────────────────────────────────────────────────────────
def test_kmmoe_softmax_over_top_k_only():
    from morie.fn.kmmoe import kamath_moe_router_softmax
    x = np.array([1.0, -0.5])
    Wr = rand_matrix(2, 5, seed=73)
    experts = [(lambda c: (lambda v: float(np.sum(v)) * c))(i + 1)
               for i in range(5)]
    out = kamath_moe_router_softmax(x, Wr, experts, 2)
    scores = x @ Wr
    top = sorted(range(5), key=lambda i: -scores[i])[:2]
    e = {i: math.exp(scores[i] - max(scores[j] for j in top)) for i in top}
    Z = sum(e.values())
    want = sum((e[i] / Z) * float(np.sum(x)) * (i + 1) for i in top)
    assert out["estimate"] == pytest.approx(want)
    assert out["selected_experts"] == sorted(top)
    assert sum(out["gate_weights"]) == pytest.approx(1.0)
    assert out["experts_evaluated"] == 2


def test_kmmoe_expert_count_must_match_router():
    from morie.fn.kmmoe import kamath_moe_router_softmax
    with pytest.raises(ValueError):
        kamath_moe_router_softmax([1.0], [[1.0, 2.0]], [lambda v: 1.0], 1)


# ── kmmsc ─────────────────────────────────────────────────────────
def test_kmmsc_wmd_equals_the_best_assignment():
    from morie.fn.kmmsc import word_movers_distance
    # With equal uniform weights the transportation optimum is attained
    # at a permutation (Birkhoff), so brute force is an exact oracle.
    H = rand_matrix(4, 2, seed=79)
    R = rand_matrix(4, 2, seed=83)
    C = np.array([[float(np.linalg.norm(h - r)) for r in R] for h in H])
    p = np.full(4, 0.25)
    got = word_movers_distance(C, p, p)
    best = min(sum(C[i, perm[i]] for i in range(4)) * 0.25
               for perm in permutations(range(4)))
    assert got == pytest.approx(best)


def test_kmmsc_score_bounds_and_identity():
    from morie.fn.kmmsc import kamath_moverscore
    H = rand_matrix(5, 3, seed=89)
    same = kamath_moverscore(H, H)
    assert same["wmd"] == pytest.approx(0.0, abs=1e-9)
    assert same["estimate"] == pytest.approx(1.0)
    other = kamath_moverscore(H, rand_matrix(5, 3, seed=97))
    assert 0.0 <= other["estimate"] <= 1.0
    assert other["wmd"] > 0


# ── kmnf4 ─────────────────────────────────────────────────────────
def test_kmnf4_levels_are_the_equal_mass_quantiles():
    from morie.fn.kmnf4 import kamath_nf4_datatype
    out = kamath_nf4_datatype(16)
    # Independent route: push each level back through the normal CDF.
    for i, q in enumerate(out["levels"]):
        cdf = 0.5 * (1 + math.erf(q / math.sqrt(2)))
        assert cdf == pytest.approx((i + 0.5) / 16, abs=1e-12)
    assert out["levels"] == pytest.approx(sorted(out["levels"]))
    assert max(abs(v) for v in out["normalized"]) == pytest.approx(1.0)


def test_kmnf4_quantile_refuses_the_endpoints():
    from morie.fn.kmnf4 import normal_quantile
    assert normal_quantile(0.5) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        normal_quantile(0.0)
    with pytest.raises(ValueError):
        normal_quantile(1.0)


# ── kmngrm ────────────────────────────────────────────────────────
def test_kmngrm_is_the_count_ratio():
    from morie.fn.kmngrm import kamath_ngram_language_model
    assert kamath_ngram_language_model(7, 20)["estimate"] == pytest.approx(0.35)
    with pytest.raises(ValueError):
        kamath_ngram_language_model(1, 0)
    with pytest.raises(ValueError):
        kamath_ngram_language_model(5, 4)


# ── kmnuc ─────────────────────────────────────────────────────────
def test_kmnuc_keeps_the_smallest_set_reaching_p():
    from morie.fn.kmnuc import kamath_nucleus_sampling
    logits = [0.0, 2.0, 1.0, -1.0]
    p = softmax(logits)
    order = np.argsort(-p)
    cum, keep = 0.0, []
    for i in order:
        keep.append(int(i))
        cum += p[i]
        if cum >= 0.8:
            break
    out = kamath_nucleus_sampling(logits, 0.8)
    assert sorted(out["kept"]) == sorted(keep)
    assert out["n_kept"] == len(keep)
    assert sum(out["probabilities"]) == pytest.approx(1.0)
    for i in range(4):
        if i not in keep:
            assert out["probabilities"][i] == 0.0
    assert out["kept_mass"] == pytest.approx(sum(p[i] for i in keep))


def test_kmnuc_rejects_bad_p_and_temperature():
    from morie.fn.kmnuc import kamath_nucleus_sampling
    with pytest.raises(ValueError):
        kamath_nucleus_sampling([1.0, 2.0], 0.0)
    with pytest.raises(ValueError):
        kamath_nucleus_sampling([1.0, 2.0], 0.5, T=0.0)


# ── kmnxtg ────────────────────────────────────────────────────────
def test_kmnxtg_runs_every_encoder_and_decoder():
    from morie.fn.kmnxtg import kamath_nextgpt_any2any
    calls = []
    out = kamath_nextgpt_any2any(
        {"text": "ab", "audio": [1, 2, 3]},
        {"text": lambda s: len(s), "audio": lambda a: sum(a)},
        lambda f: f["text"] + f["audio"],
        {"image": lambda h: ("img", h), "text": lambda h: ("txt", h)})
    assert out["llm_state"] == 8
    assert out["outputs"] == {"image": ("img", 8), "text": ("txt", 8)}
    assert out["input_modalities"] == ["audio", "text"]
    assert calls == []


def test_kmnxtg_missing_encoder_refused():
    from morie.fn.kmnxtg import kamath_nextgpt_any2any
    with pytest.raises(ValueError):
        kamath_nextgpt_any2any({"text": "a", "video": 1},
                               {"text": lambda s: 1},
                               lambda f: 0, {"text": lambda h: h})


# ── kmp2 ──────────────────────────────────────────────────────────
def test_kmp2_prepends_a_prefix_at_every_layer():
    from morie.fn.kmp2 import kamath_p_tuning_v2
    pre = [(rand_matrix(2, 3, seed=101), rand_matrix(2, 3, seed=103)),
           (rand_matrix(1, 3, seed=107), rand_matrix(1, 3, seed=109))]
    inp = [(rand_matrix(4, 3, seed=113), rand_matrix(4, 3, seed=127)),
           (rand_matrix(4, 3, seed=131), rand_matrix(4, 3, seed=137))]
    out = kamath_p_tuning_v2(pre, inp)
    assert [len(k) for k in out["K"]] == [6, 5]
    assert np.allclose(out["K"][0][:2], pre[0][0])
    assert np.allclose(out["K"][0][2:], inp[0][0])
    assert out["prefix_len"] == [2, 1]
    assert out["n_trainable"] == 2 * 3 * 2 + 1 * 3 * 2


def test_kmp2_layer_count_must_match():
    from morie.fn.kmp2 import kamath_p_tuning_v2
    with pytest.raises(ValueError):
        kamath_p_tuning_v2([([[1.0]], [[1.0]])],
                           [([[1.0]], [[1.0]]), ([[1.0]], [[1.0]])])


# ── kmpask ────────────────────────────────────────────────────────
def test_kmpask_matches_brute_force_enumeration():
    from morie.fn.kmpask import kamath_pass_at_k
    n, c, k = 7, 3, 3
    draws = list(combinations(range(n), k))
    correct = set(range(c))
    want = sum(1 for d in draws if correct & set(d)) / len(draws)
    assert kamath_pass_at_k(n, c, k)["estimate"] == pytest.approx(want)


def test_kmpask_edges_and_large_n_no_overflow():
    from morie.fn.kmpask import kamath_pass_at_k
    assert kamath_pass_at_k(200, 0, 10)["estimate"] == 0.0
    assert kamath_pass_at_k(200, 200, 10)["estimate"] == 1.0
    big = kamath_pass_at_k(2000, 1, 100)["estimate"]
    assert big == pytest.approx(0.05, abs=1e-9)
    with pytest.raises(ValueError):
        kamath_pass_at_k(5, 6, 1)


# ── kmperp ────────────────────────────────────────────────────────
def test_kmperp_uniform_model_has_perplexity_v():
    from morie.fn.kmperp import kamath_perplexity
    V = 7
    lp = [-math.log(V)] * 10
    assert kamath_perplexity(lp)["estimate"] == pytest.approx(V)
    mixed = [-0.5, -1.5, -0.25]
    want = math.exp(-sum(mixed) / 3)
    assert kamath_perplexity(mixed)["estimate"] == pytest.approx(want)
    assert kamath_perplexity(mixed)["estimate"] != pytest.approx(np.mean(mixed))


def test_kmperp_positive_logprob_refused():
    from morie.fn.kmperp import kamath_perplexity
    with pytest.raises(ValueError):
        kamath_perplexity([0.5, -0.5])
    with pytest.raises(ValueError):
        kamath_perplexity([])


# ── kmpet ─────────────────────────────────────────────────────────
def test_kmpet_adds_the_weighted_mlm_term():
    from morie.fn.kmpet import kamath_pet_loss
    vz = [0.4, -1.0, 2.0]
    ml = rand_matrix(4, 5, seed=139)
    mt = [-100, 2, -100, 4]
    alpha = 0.3
    ce = -math.log(softmax(vz)[2])
    mlm = np.mean([-math.log(softmax(ml[1])[2]), -math.log(softmax(ml[3])[4])])
    out = kamath_pet_loss(vz, 2, ml, mt, alpha)
    assert out["loss_ce"] == pytest.approx(ce)
    assert out["loss_mlm"] == pytest.approx(mlm)
    assert out["estimate"] == pytest.approx(ce + alpha * mlm)
    assert out["n_masked"] == 2


def test_kmpet_needs_a_mask_and_a_non_negative_alpha():
    from morie.fn.kmpet import kamath_pet_loss
    with pytest.raises(ValueError):
        kamath_pet_loss([0.0, 0.0], 0, [[0.0, 0.0]], [-100], 1.0)
    with pytest.raises(ValueError):
        kamath_pet_loss([0.0, 0.0], 0, [[0.0, 0.0]], [1], -0.5)


# ── kmpoln / kmprln ───────────────────────────────────────────────
def _ln(v, eps=1e-5):
    v = np.asarray(v, dtype=float)
    return (v - v.mean(-1, keepdims=True)) / np.sqrt(
        v.var(-1, keepdims=True) + eps)


def test_kmpoln_normalises_after_the_residual_add():
    from morie.fn.kmpoln import kamath_post_ln_transformer
    x = rand_matrix(2, 4, seed=149)
    A = rand_matrix(2, 4, seed=151)
    F = rand_matrix(2, 4, seed=157)
    out = kamath_post_ln_transformer(x, lambda v: A, lambda v: F)
    y = _ln(x + A)
    want = _ln(y + F)
    assert np.allclose(out["output"], want)
    # Post-LN output rows are standardised; pre-LN's are not.
    assert np.allclose(np.mean(out["output"], axis=1), 0.0, atol=1e-8)


def test_kmprln_keeps_a_clean_residual_path():
    from morie.fn.kmprln import kamath_pre_ln_transformer
    from morie.fn.kmpoln import kamath_post_ln_transformer
    x = rand_matrix(2, 4, seed=149)
    A = rand_matrix(2, 4, seed=151)
    F = rand_matrix(2, 4, seed=157)
    out = kamath_pre_ln_transformer(x, lambda v: A, lambda v: F)
    assert np.allclose(out["output"], x + A + F)
    post = kamath_post_ln_transformer(x, lambda v: A, lambda v: F)
    assert not np.allclose(out["output"], post["output"])


def test_prepost_ln_reject_shape_changing_sublayers():
    from morie.fn.kmprln import kamath_pre_ln_transformer
    with pytest.raises(ValueError):
        kamath_pre_ln_transformer([[1.0, 2.0]], lambda v: [[1.0]],
                                  lambda v: v)


# ── kmppok ────────────────────────────────────────────────────────
def test_kmppok_penalises_the_log_ratio():
    from morie.fn.kmppok import kamath_ppo_rlhf_objective
    r = [1.0, 2.0, -0.5]
    lt = [math.log(0.4), math.log(0.5), math.log(0.1)]
    lr = [math.log(0.2), math.log(0.5), math.log(0.4)]
    beta = 0.3
    want = np.mean([a - beta * (b - c) for a, b, c in zip(r, lt, lr)])
    out = kamath_ppo_rlhf_objective(r, lt, lr, beta)
    assert out["estimate"] == pytest.approx(want)
    assert out["kl_estimate"] == pytest.approx(np.mean(np.subtract(lt, lr)))
    assert out["estimate"] != pytest.approx(np.mean(r))


def test_kmppok_rejects_positive_logprobs():
    from morie.fn.kmppok import kamath_ppo_rlhf_objective
    with pytest.raises(ValueError):
        kamath_ppo_rlhf_objective([1.0], [0.5], [-0.5], 1.0)


# ── kmpref ────────────────────────────────────────────────────────
def test_kmpref_extends_keys_and_values_and_attends_over_both():
    from morie.fn.kmpref import kamath_prefix_tuning
    PK, PV = rand_matrix(2, 3, seed=163), rand_matrix(2, 5, seed=167)
    K, V = rand_matrix(4, 3, seed=173), rand_matrix(4, 5, seed=179)
    Q = rand_matrix(1, 3, seed=181)
    out = kamath_prefix_tuning(PK, PV, K, V, Q=Q)
    assert np.allclose(out["K"], np.vstack([PK, K]))
    assert np.allclose(out["V"], np.vstack([PV, V]))
    scores = Q @ np.vstack([PK, K]).T / math.sqrt(3)
    w = softmax(scores[0])
    assert np.allclose(out["attention_output"][0], w @ np.vstack([PV, V]))
    assert out["prefix_attention_mass"][0] == pytest.approx(w[:2].sum())


def test_kmpref_empty_prefix_refused():
    from morie.fn.kmpref import kamath_prefix_tuning
    with pytest.raises(ValueError):
        kamath_prefix_tuning(np.zeros((0, 3)), np.zeros((0, 2)),
                             [[1.0, 0.0, 0.0]], [[1.0, 1.0]])


# ── kmptun ────────────────────────────────────────────────────────
def test_kmptun_prepends_the_soft_prompt():
    from morie.fn.kmptun import kamath_prompt_tuning
    P = rand_matrix(3, 6, seed=191)
    X = rand_matrix(5, 6, seed=193)
    out = kamath_prompt_tuning(P, X)
    assert np.allclose(out["X_aug"][:3], P)
    assert np.allclose(out["X_aug"][3:], X)
    assert out["n_trainable"] == 18 and out["seq_len"] == 8


def test_kmptun_dimension_mismatch_refused():
    from morie.fn.kmptun import kamath_prompt_tuning
    with pytest.raises(ValueError):
        kamath_prompt_tuning([[1.0, 2.0]], [[1.0, 2.0, 3.0]])


# ── kmqfrm ────────────────────────────────────────────────────────
def test_kmqfrm_is_cross_attention_from_queries_to_patches():
    from morie.fn.kmqfrm import kamath_q_former
    Q = rand_matrix(2, 4, seed=197)
    F = rand_matrix(7, 4, seed=199)
    out = kamath_q_former(Q, F)
    S = Q @ F.T / math.sqrt(4)
    want = np.vstack([softmax(S[i]) @ F for i in range(2)])
    assert np.allclose(out["Z"], want)
    assert np.allclose(np.sum(out["attention"], axis=1), 1.0)
    assert out["compression"] == pytest.approx(3.5)


def test_kmqfrm_width_mismatch_refused():
    from morie.fn.kmqfrm import kamath_q_former
    with pytest.raises(ValueError):
        kamath_q_former([[1.0, 0.0]], [[1.0, 0.0, 0.0]])


# ── kmqlor ────────────────────────────────────────────────────────
def test_kmqlor_dequantises_then_applies_the_adapter():
    from morie.fn.kmqlor import kamath_qlora_4bit, dequantize_nf4
    from morie.fn.kmlora import kamath_lora_weight_update
    from morie.fn.kmnf4 import kamath_nf4_datatype
    codes = [[0, 7, 15], [8, 3, 12]]
    absmax = 0.75
    W0 = dequantize_nf4(codes, absmax)
    grid = kamath_nf4_datatype(16)["normalized"]
    assert np.allclose(W0, [[grid[c] * absmax for c in row] for row in codes])
    A = rand_matrix(1, 3, seed=211)
    B = rand_matrix(2, 1, seed=223)
    x = rand_matrix(1, 3, seed=227)[0]
    q = kamath_qlora_4bit({"codes": codes, "absmax": absmax}, A, B,
                          4.0, 1, x)
    plain = kamath_lora_weight_update(W0, A, B, 4.0, 1, x)
    assert q["h"] == pytest.approx(plain["h"])


def test_kmqlor_rejects_out_of_range_codes():
    from morie.fn.kmqlor import kamath_qlora_4bit
    with pytest.raises(ValueError):
        kamath_qlora_4bit({"codes": [[16]], "absmax": 1.0}, [[1.0]],
                          [[1.0]], 1.0, 1, [1.0])


# ── kmret ─────────────────────────────────────────────────────────
def test_kmret_parallel_form_equals_the_recurrence():
    from morie.fn.kmret import kamath_retnet_retention
    Q = rand_matrix(6, 3, seed=229)
    K = rand_matrix(6, 3, seed=233)
    V = rand_matrix(6, 2, seed=239)
    out = kamath_retnet_retention(Q, K, V, 0.9)
    # Independent route: the O(1) recurrent form.
    S = np.zeros((3, 2))
    want = []
    for t in range(6):
        S = 0.9 * S + np.outer(K[t], V[t])
        want.append(Q[t] @ S)
    assert np.allclose(out["output"], want)
    D = np.asarray(out["decay"])
    assert np.allclose(np.triu(D, 1), 0.0)      # strictly causal
    assert D[3, 1] == pytest.approx(0.9 ** 2)


def test_kmret_rejects_bad_gamma():
    from morie.fn.kmret import kamath_retnet_retention
    with pytest.raises(ValueError):
        kamath_retnet_retention([[1.0]], [[1.0]], [[1.0]], 1.5)


# ── kmrhf ─────────────────────────────────────────────────────────
def test_kmrhf_anchors_ppo_to_the_sft_policy():
    from morie.fn.kmrhf import kamath_rlhf_pipeline
    seen = {}

    def ppo(pi, rm, ref):
        seen["pi"], seen["ref"] = pi, ref
        return {"from": pi}

    out = kamath_rlhf_pipeline(["d"], [("a", "b")], "base",
                               sft=lambda p, d: {"sft": p},
                               train_rm=lambda p: (lambda y: 1.0),
                               ppo=ppo)
    assert seen["pi"] is seen["ref"]
    assert seen["pi"] is out["policy_sft"]
    assert seen["pi"] != "base"
    assert out["kl_reference_is_sft"] is True


def test_kmrhf_requires_all_three_stages_and_data():
    from morie.fn.kmrhf import kamath_rlhf_pipeline
    with pytest.raises(ValueError):
        kamath_rlhf_pipeline(["d"], [("a", "b")], "p", sft=lambda p, d: p)
    with pytest.raises(ValueError):
        kamath_rlhf_pipeline([], [("a", "b")], "p", sft=lambda p, d: p,
                             train_rm=lambda p: (lambda y: 1.0),
                             ppo=lambda a, b, c: a)


# ── kmrlaif ───────────────────────────────────────────────────────
def test_kmrlaif_satisfies_the_bradley_terry_stationarity_condition():
    from morie.fn.kmrlaif import kamath_rlaif_objective
    prefs = [(0, 1), (0, 1), (1, 2), (2, 0), (2, 1), (1, 0)]
    out = kamath_rlaif_objective(prefs)
    p = np.array(out["strengths"])
    items = out["items"]
    idx = {it: i for i, it in enumerate(items)}
    wins = np.zeros(3)
    counts = np.zeros((3, 3))
    for w, l in prefs:
        wins[idx[w]] += 1
        counts[idx[w], idx[l]] += 1
        counts[idx[l], idx[w]] += 1
    # MLE first-order condition: w_i == sum_j n_ij p_i / (p_i + p_j).
    for i in range(3):
        rhs = sum(counts[i, j] * p[i] / (p[i] + p[j])
                  for j in range(3) if j != i)
        assert rhs == pytest.approx(wins[i], abs=1e-6)
    assert sum(p) == pytest.approx(1.0)


def test_kmrlaif_refuses_a_disconnected_graph():
    from morie.fn.kmrlaif import kamath_rlaif_objective
    with pytest.raises(ValueError):
        kamath_rlaif_objective([(0, 1), (0, 2)])     # 0 never loses


# ── kmrmloss ──────────────────────────────────────────────────────
def test_kmrmloss_is_invariant_to_a_constant_shift():
    from morie.fn.kmrmloss import kamath_reward_model_training_loss
    w = [2.0, -1.0, 0.5]
    l = [1.0, 0.5, 0.5]
    base = kamath_reward_model_training_loss(w, l)
    shifted = kamath_reward_model_training_loss(
        [v + 100 for v in w], [v + 100 for v in l])
    assert shifted["estimate"] == pytest.approx(base["estimate"])
    want = np.mean([-math.log(1 / (1 + math.exp(-(a - b))))
                    for a, b in zip(w, l)])
    assert base["estimate"] == pytest.approx(want)
    assert base["accuracy"] == pytest.approx(1 / 3)


def test_kmrmloss_does_not_overflow_on_a_large_margin():
    from morie.fn.kmrmloss import kamath_reward_model_training_loss
    out = kamath_reward_model_training_loss([800.0], [-800.0])
    assert out["estimate"] == 0.0
    assert np.isfinite(out["estimate"])


# ── kmrmsn ────────────────────────────────────────────────────────
def test_kmrmsn_divides_by_the_root_mean_square():
    from morie.fn.kmrmsn import kamath_rms_norm
    x = [1.0, -2.0, 3.0, 4.0]
    out = kamath_rms_norm(x, eps=0.0)
    rms = math.sqrt(sum(v * v for v in x) / 4)
    assert out["y"] == pytest.approx([v / rms for v in x])
    # RMSNorm does NOT centre: the mean of the output is not 0 here.
    assert abs(np.mean(out["y"])) > 1e-6
    g = [1.0, 2.0, 3.0, 4.0]
    scaled = kamath_rms_norm(x, g=g, eps=0.0)
    assert scaled["y"] == pytest.approx([v / rms * gg for v, gg in zip(x, g)])


def test_kmrmsn_zero_input_with_zero_eps_refused():
    from morie.fn.kmrmsn import kamath_rms_norm
    with pytest.raises(ValueError):
        kamath_rms_norm([0.0, 0.0], eps=0.0)
    with pytest.raises(ValueError):
        kamath_rms_norm([1.0, 2.0], g=[1.0])


# ── kmrope ────────────────────────────────────────────────────────
def test_kmrope_agrees_with_rotrp_on_contiguous_positions():
    from morie.fn.kmrope import kamath_rotary_positional_embedding
    from morie.fn.rotrp import rotary_position_embedding
    x = rand_matrix(5, 8, seed=241)
    mine = kamath_rotary_positional_embedding(x, list(range(5)))
    theirs = rotary_position_embedding(x)
    assert np.allclose(mine["y"], theirs["y"])


def test_kmrope_is_a_rotation_and_honours_explicit_positions():
    from morie.fn.kmrope import kamath_rotary_positional_embedding
    x = rand_matrix(3, 6, seed=251)
    out = kamath_rotary_positional_embedding(x, [17, 0, 4])
    y = np.asarray(out["y"])
    # Rotations preserve the norm of every feature pair.
    for i in range(3):
        for j in range(0, 6, 2):
            assert (y[i, j] ** 2 + y[i, j + 1] ** 2) == pytest.approx(
                x[i, j] ** 2 + x[i, j + 1] ** 2)
    assert np.allclose(y[1], x[1])              # position 0 = identity
    with pytest.raises(ValueError):
        kamath_rotary_positional_embedding([[1.0, 2.0, 3.0]], [0])


# ── kmroug ────────────────────────────────────────────────────────
def test_kmroug_clips_counts_at_the_reference():
    from morie.fn.kmroug import kamath_rouge_n
    hyp, ref = "a a a b", "a b c a"
    out = kamath_rouge_n(hyp, ref, 1)
    hc, rc = Counter(hyp.split()), Counter(ref.split())
    want = sum(min(c, hc[g]) for g, c in rc.items()) / sum(rc.values())
    assert out["estimate"] == pytest.approx(want) == pytest.approx(3 / 4)
    assert out["precision"] == pytest.approx(3 / 4)


def test_kmroug_bigrams_and_bad_input():
    from morie.fn.kmroug import kamath_rouge_n
    out = kamath_rouge_n(["a", "b", "c"], ["a", "b", "c", "d"], 2)
    assert out["estimate"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        kamath_rouge_n("a b", "a", 2)
    with pytest.raises(ValueError):
        kamath_rouge_n("a", "", 1)


# ── kmrrf ─────────────────────────────────────────────────────────
def test_kmrrf_sums_reciprocal_ranks_only_where_seen():
    from morie.fn.kmrrf import kamath_reciprocal_rank_fusion
    rankings = [["a", "b", "c"], ["c", "a"], ["b", "c", "a"]]
    out = kamath_reciprocal_rank_fusion(rankings, k=10)
    want = {}
    for r in rankings:
        for pos, d in enumerate(r, 1):
            want[d] = want.get(d, 0.0) + 1 / (10 + pos)
    for d, v in want.items():
        assert out["scores"][d] == pytest.approx(v)
    assert out["ranking"] == sorted(want, key=lambda d: (-want[d], d))
    assert out["appearances"] == {"a": 3, "b": 2, "c": 3}


def test_kmrrf_rejects_empty_and_duplicated_rankings():
    from morie.fn.kmrrf import kamath_reciprocal_rank_fusion
    with pytest.raises(ValueError):
        kamath_reciprocal_rank_fusion([[]])
    with pytest.raises(ValueError):
        kamath_reciprocal_rank_fusion([["a", "a"]])


# ── kmrsft ────────────────────────────────────────────────────────
def test_kmrsft_selects_per_prompt_not_globally():
    from morie.fn.kmrsft import kamath_rejection_sampling_finetune
    prompts = ["easy", "hard"]
    samples = [["e1", "e2", "e3"], ["h1", "h2", "h3"]]
    rewards = [[9.0, 8.0, 7.0], [1.0, 0.5, 2.0]]
    out = kamath_rejection_sampling_finetune(prompts, samples, rewards, 1)
    assert out["retained"] == [("easy", "e1"), ("hard", "h3")]
    assert out["n_retained"] == 2 and out["n_dropped"] == 4
    # A global top-2 would have kept only the easy prompt's samples.
    assert {p for p, _ in out["retained"]} == {"easy", "hard"}


def test_kmrsft_optional_sft_hook_and_bad_shapes():
    from morie.fn.kmrsft import kamath_rejection_sampling_finetune
    out = kamath_rejection_sampling_finetune(
        ["p"], [["a", "b"]], [[1.0, 2.0]], 5, sft=lambda pairs: len(pairs))
    assert out["policy"] == 2            # k clipped to what exists
    with pytest.raises(ValueError):
        kamath_rejection_sampling_finetune(["p"], [["a"]], [[1.0, 2.0]], 1)


# ── kmrwkv ────────────────────────────────────────────────────────
def test_kmrwkv_matches_the_naive_direct_sum():
    from morie.fn.kmrwkv import kamath_rwkv_time_mix
    k = [0.3, -0.7, 1.2, 0.0]
    v = [1.0, -2.0, 0.5, 4.0]
    w, u = 0.6, 0.25
    out = kamath_rwkv_time_mix(k, v, w, u)
    want = []
    for t in range(4):
        num = den = 0.0
        for i in range(t + 1):
            e = math.exp(-(t - i) * w + k[i] + (u if i == t else 0.0))
            num += e * v[i]
            den += e
        want.append(num / den)
    assert out["wkv"] == pytest.approx(want)
    # It is a weighted average, so it lies inside the value range.
    assert min(v) <= min(out["wkv"]) and max(out["wkv"]) <= max(v)


def test_kmrwkv_stable_at_extreme_k_and_rejects_negative_w():
    from morie.fn.kmrwkv import kamath_rwkv_time_mix
    out = kamath_rwkv_time_mix([900.0, 900.0, 900.0], [1.0, 2.0, 3.0], 0.0)
    assert out["wkv"][2] == pytest.approx(2.0)
    assert all(np.isfinite(out["wkv"]))
    with pytest.raises(ValueError):
        kamath_rwkv_time_mix([1.0], [1.0], -0.5)


# ── kmsc ──────────────────────────────────────────────────────────
def test_kmsc_majority_vote_over_parsed_answers():
    from morie.fn.kmsc import kamath_self_consistency
    samples = ["... so 7", "... so 4", "... so 7", "... so 7", "... so 4"]
    out = kamath_self_consistency(samples, parse=lambda s: s.split()[-1])
    counts = Counter(s.split()[-1] for s in samples)
    assert out["answer"] == counts.most_common(1)[0][0] == "7"
    assert out["votes"] == 3
    assert out["agreement"] == pytest.approx(3 / 5)
    assert out["tie"] is False


def test_kmsc_unparsed_traces_excluded_not_voted():
    from morie.fn.kmsc import kamath_self_consistency
    out = kamath_self_consistency(["a", "bad", "a"],
                                  parse=lambda s: None if s == "bad" else s)
    assert out["n_unparsed"] == 1 and out["n_voted"] == 2
    assert out["answer"] == "a" and out["agreement"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        kamath_self_consistency(["x"], parse=lambda s: None)


# ── kmscal ────────────────────────────────────────────────────────
def test_kmscal_is_a_power_law_with_a_floor():
    from morie.fn.kmscal import kamath_scaling_laws
    N = [1e8, 1e9, 1e10]
    out = kamath_scaling_laws(N, 8.8e13, 0.076, 1.69)
    want = [(8.8e13 / n) ** 0.076 + 1.69 for n in N]
    assert out["loss"] == pytest.approx(want)
    # Ten times the parameters multiplies the reducible part by 10^-alpha.
    assert out["reducible"][1] / out["reducible"][0] == pytest.approx(
        10 ** -0.076)
    assert all(out["loss"][i] > out["loss"][i + 1] for i in range(2))


def test_kmscal_rejects_degenerate_parameters():
    from morie.fn.kmscal import kamath_scaling_laws
    with pytest.raises(ValueError):
        kamath_scaling_laws(0.0, 1e3, 0.1)
    with pytest.raises(ValueError):
        kamath_scaling_laws(1e3, 1e3, -0.1)


# ── kmsp / kmuni ──────────────────────────────────────────────────
def test_kmuni_em_never_decreases_the_log_likelihood():
    from morie.fn.kmuni import kamath_unigram_lm_tokenizer, unigram_loglik
    corpus = ["abab", "abc", "cab", "ab"]
    out = kamath_unigram_lm_tokenizer(corpus, ["a", "b", "c", "ab", "ca"])
    hist = out["log_likelihood_history"]
    assert all(b >= a - 1e-12 for a, b in zip(hist, hist[1:]))
    assert out["log_likelihood"] == pytest.approx(
        unigram_loglik(corpus, out["probs"]))
    assert sum(out["probs"].values()) == pytest.approx(1.0)
    # A forced segmentation gives exactly the relative frequencies.
    forced = kamath_unigram_lm_tokenizer(["ab", "a", "b", "b"], ["a", "b"])
    assert forced["probs"]["a"] == pytest.approx(2 / 5)
    assert forced["probs"]["b"] == pytest.approx(3 / 5)


def test_kmuni_viterbi_beats_or_matches_every_other_segmentation():
    from morie.fn.kmuni import viterbi_segment
    probs = {"a": 0.1, "b": 0.2, "ab": 0.5, "aba": 0.05, "c": 0.15}
    seg, score = viterbi_segment("abab", probs)
    assert seg == ["ab", "ab"]
    assert score == pytest.approx(2 * math.log(0.5))
    # Enumerate all segmentations of a short string and compare.
    def enumerate_segs(s):
        if not s:
            yield []
            return
        for n in range(1, len(s) + 1):
            if s[:n] in probs:
                for rest in enumerate_segs(s[n:]):
                    yield [s[:n]] + rest
    best = max(sum(math.log(probs[w]) for w in seg2)
               for seg2 in enumerate_segs("abab"))
    assert score == pytest.approx(best)


def test_kmsp_keeps_every_character_and_hits_the_target_size():
    from morie.fn.kmsp import kamath_sentencepiece_tokenizer
    from morie.fn.kmuni import unigram_loglik
    corpus = ["hello world", "hello there", "world world"]
    out = kamath_sentencepiece_tokenizer(corpus, 14)
    assert out["vocab_size"] == 14
    assert set(c for s in corpus for c in s) <= set(out["vocab"])
    assert out["log_likelihood"] == pytest.approx(
        unigram_loglik(corpus, out["probs"]))
    # Every sentence still segments, and the pieces reconstruct it.
    for s, seg in zip(corpus, out["segmentations"]):
        assert "".join(seg) == s


def test_kmsp_refuses_a_vocab_below_the_character_count():
    from morie.fn.kmsp import kamath_sentencepiece_tokenizer
    with pytest.raises(ValueError):
        kamath_sentencepiece_tokenizer(["abcdef"], 3)


# ── kmspd ─────────────────────────────────────────────────────────
def test_kmspd_accept_plus_residual_reproduces_the_target():
    from morie.fn.kmspd import kamath_speculative_decoding
    pd = np.array([0.5, 0.3, 0.2])
    pt = np.array([0.2, 0.5, 0.3])
    # The losslessness identity: for every token,
    #   p_draft(t)*min(1, pt/pd) + P(reject)*residual(t) == p_target(t).
    accept = {t: kamath_speculative_decoding(pd, pt, proposed=t)
              for t in range(3)}
    p_reject = 1.0 - sum(pd[t] * accept[t]["accept_prob"] for t in range(3))
    resid = np.array(accept[0]["residual"])
    for t in range(3):
        got = pd[t] * accept[t]["accept_prob"] + p_reject * resid[t]
        assert got == pytest.approx(pt[t])
    assert accept[0]["rejection_rate"] == pytest.approx(p_reject)


def test_kmspd_uniform_draw_decides_and_bad_input_refused():
    from morie.fn.kmspd import kamath_speculative_decoding
    a = kamath_speculative_decoding([0.5, 0.5], [0.25, 0.75], proposed=0,
                                    u=0.49)
    b = kamath_speculative_decoding([0.5, 0.5], [0.25, 0.75], proposed=0,
                                    u=0.51)
    assert a["accepted"] is True and b["accepted"] is False
    with pytest.raises(ValueError):
        kamath_speculative_decoding([0.5, 0.4], [0.5, 0.5])


# ── kmspn ─────────────────────────────────────────────────────────
def test_kmspn_input_and_target_partition_the_sequence():
    from morie.fn.kmspn import kamath_t5_span_corruption
    toks = [f"t{i}" for i in range(20)]
    out = kamath_t5_span_corruption(toks, 3.0, 0.15, seed=5)
    kept = [t for t in out["input"] if not t.startswith("<extra_id")]
    masked = [t for t in out["target"] if not t.startswith("<extra_id")]
    assert sorted(kept + masked, key=lambda t: int(t[1:])) == toks
    assert len(masked) == out["n_masked"] == 3
    assert len(out["spans"]) == out["n_spans"] == 1
    assert len(out["input"]) < len(toks)              # spans compress


def test_kmspn_is_deterministic_per_seed():
    from morie.fn.kmspn import kamath_t5_span_corruption
    toks = list("abcdefghijklmnop")
    a = kamath_t5_span_corruption(toks, 2.0, 0.5, seed=3)
    b = kamath_t5_span_corruption(toks, 2.0, 0.5, seed=3)
    c = kamath_t5_span_corruption(toks, 2.0, 0.5, seed=4)
    assert a["input"] == b["input"] and a["target"] == b["target"]
    assert (a["input"], a["span_lengths"]) != (c["input"], c["span_lengths"])
    with pytest.raises(ValueError):
        kamath_t5_span_corruption(toks, 2.0, 1.0)


# ── kmsrag ────────────────────────────────────────────────────────
def test_kmsrag_decodes_each_reflection_group():
    from morie.fn.kmsrag import kamath_self_rag
    out = kamath_self_rag(
        ["c"], lambda c, q: "[Retrieve] [Irrelevant] [No Support] "
                            "[Utility:4]")
    assert out["retrieve"] is True
    assert out["relevant"] is False
    assert out["supported"] is False
    assert out["support_level"] == "[No Support]"
    assert out["utility"] == 4


def test_kmsrag_unknown_and_contradictory_tokens_refused():
    from morie.fn.kmsrag import kamath_self_rag
    with pytest.raises(ValueError):
        kamath_self_rag(["c"], lambda c, q: ["[Maybe]"])
    with pytest.raises(ValueError):
        kamath_self_rag(["c"], lambda c, q: ["[Relevant]", "[Irrelevant]"])
    with pytest.raises(ValueError):
        kamath_self_rag(["c"], lambda c, q: [])


# ── kmstb ─────────────────────────────────────────────────────────
def test_kmstb_unions_both_contexts_without_duplicates():
    from morie.fn.kmstb import kamath_step_back_prompting
    docs = {"general": ["g1", "shared"], "specific": ["shared", "s1"]}
    out = kamath_step_back_prompting(
        "specific", lambda q: "general", retrieve=docs.get)
    assert out["context"] == ["g1", "shared", "s1"]   # abstraction first
    assert out["n_context"] == 3
    assert out["stepped_back"] is True


def test_kmstb_flags_a_model_that_did_not_step_back():
    from morie.fn.kmstb import kamath_step_back_prompting
    out = kamath_step_back_prompting("q", lambda q: "q")
    assert out["stepped_back"] is False
    assert "warning" in out
    with pytest.raises(ValueError):
        kamath_step_back_prompting("q", lambda q: None)


# ── kmstgn ────────────────────────────────────────────────────────
def test_kmstgn_matches_its_two_delegates_run_separately():
    from morie.fn.kmstgn import kamath_summarize_from_feedback
    from morie.fn.kmrmloss import kamath_reward_model_training_loss
    from morie.fn.kmppok import kamath_ppo_rlhf_objective
    prefs = [(1.5, 0.5), (0.2, 0.9), (3.0, -1.0)]
    rewards = [0.5, 1.5, -0.5]
    lt = [math.log(0.3), math.log(0.6), math.log(0.1)]
    lr = [math.log(0.2), math.log(0.7), math.log(0.1)]
    out = kamath_summarize_from_feedback(prefs, rewards, lt, lr, 0.25)
    rm = kamath_reward_model_training_loss([p[0] for p in prefs],
                                           [p[1] for p in prefs])
    rl = kamath_ppo_rlhf_objective(rewards, lt, lr, 0.25)
    assert out["loss_rm"] == pytest.approx(rm["estimate"])
    assert out["objective"] == pytest.approx(rl["estimate"])
    assert out["rm_accuracy"] == pytest.approx(2 / 3)


def test_kmstgn_rejects_malformed_preferences():
    from morie.fn.kmstgn import kamath_summarize_from_feedback
    with pytest.raises(ValueError):
        kamath_summarize_from_feedback([(1.0, 2.0, 3.0)], [1.0], [-1.0],
                                       [-1.0], 0.1)


# ── kmstst ────────────────────────────────────────────────────────
def test_kmstst_counts_stereotype_preferences_ties_apart():
    from morie.fn.kmstst import kamath_stereoset_bias
    s = [0.9, 0.1, 0.5, 0.7, 0.2]
    a = [0.1, 0.9, 0.5, 0.3, 0.8]
    out = kamath_stereoset_bias(s, a)
    want = sum(1 for x, y in zip(s, a) if x > y) / len(s)
    assert out["estimate"] == pytest.approx(want) == pytest.approx(0.4)
    assert out["n_ties"] == 1
    assert out["n_anti_preferred"] == 2
    assert out["bias_magnitude"] == pytest.approx(0.1)


def test_kmstst_length_mismatch_refused():
    from morie.fn.kmstst import kamath_stereoset_bias
    with pytest.raises(ValueError):
        kamath_stereoset_bias([0.5, 0.5], [0.5])


# ── kmswig ────────────────────────────────────────────────────────
def test_kmswig_gates_one_projection_with_the_swish_of_the_other():
    from morie.fn.kmswig import kamath_swiglu_activation
    x = rand_matrix(1, 3, seed=257)[0]
    W = rand_matrix(3, 4, seed=263)
    V = rand_matrix(3, 4, seed=269)
    b = rand_matrix(1, 4, seed=271)[0]
    c = rand_matrix(1, 4, seed=277)[0]
    out = kamath_swiglu_activation(x, W, V, b, c)
    g = x @ W + b
    want = (g / (1 + np.exp(-g))) * (x @ V + c)
    assert out["output"] == pytest.approx(list(want))
    # It is NOT symmetric in W and V.
    flipped = kamath_swiglu_activation(x, V, W, c, b)
    assert not np.allclose(out["output"], flipped["output"])


def test_kmswig_shape_mismatch_refused_and_swish_is_stable():
    from morie.fn.kmswig import kamath_swiglu_activation, swish
    assert float(swish(np.array(-800.0))) == pytest.approx(0.0)
    assert np.isfinite(float(swish(np.array(-800.0))))
    with pytest.raises(ValueError):
        kamath_swiglu_activation([1.0], [[1.0, 1.0]], [[1.0]])


# ── kmtemp ────────────────────────────────────────────────────────
def test_kmtemp_sharpens_and_flattens_monotonically():
    from morie.fn.kmtemp import kamath_temperature_sampling
    z = [2.0, 1.0, -1.0, 0.5]
    hot = kamath_temperature_sampling(z, 5.0)
    unit = kamath_temperature_sampling(z, 1.0)
    cold = kamath_temperature_sampling(z, 0.2)
    assert unit["probabilities"] == pytest.approx(list(softmax(z)))
    assert cold["entropy"] < unit["entropy"] < hot["entropy"]
    assert hot["entropy"] <= unit["max_entropy"] + 1e-12
    assert cold["estimate"] > unit["estimate"] > hot["estimate"]
    assert all(o["argmax"] == 0 for o in (hot, unit, cold))


def test_kmtemp_zero_temperature_refused():
    from morie.fn.kmtemp import kamath_temperature_sampling
    with pytest.raises(ValueError):
        kamath_temperature_sampling([1.0, 2.0], 0.0)
    with pytest.raises(ValueError):
        kamath_temperature_sampling([1.0, 2.0], -1.0)


# ── kmtopk ────────────────────────────────────────────────────────
def test_kmtopk_renormalises_rather_than_softmaxing():
    from morie.fn.kmtopk import kamath_moe_top_k_gating
    from morie.fn.km040 import kamath_ch2_moe_topk_gating
    g = [0.4, 0.1, 0.3, 0.2]
    out = kamath_moe_top_k_gating(g, 2)
    assert out["weights"] == pytest.approx([0.4 / 0.7, 0.0, 0.3 / 0.7, 0.0])
    assert sum(out["weights"]) == pytest.approx(1.0)
    assert out["selected_experts"] == [0, 2]
    assert out["kept_mass"] == pytest.approx(0.7)
    # The masked-softmax sibling gives DIFFERENT weights from the same
    # scores -- the two must not be confused.
    other = kamath_ch2_moe_topk_gating([1.0], [g], k=2)
    assert not np.allclose(out["weights"], other["weights"])


def test_kmtopk_negative_gates_refused():
    from morie.fn.kmtopk import kamath_moe_top_k_gating
    with pytest.raises(ValueError):
        kamath_moe_top_k_gating([0.5, -0.1], 1)
    with pytest.raises(ValueError):
        kamath_moe_top_k_gating([0.5, 0.5], 3)


# ── kmtot ─────────────────────────────────────────────────────────
def test_kmtot_beam_search_matches_exhaustive_search():
    from morie.fn.kmtot import kamath_tree_of_thoughts
    table = {"": [("a", 1.0), ("b", 3.0)],
             "a": [("aa", 9.0), ("ab", 0.5)],
             "b": [("ba", 1.0), ("bb", 2.0)]}

    def model(state, b):
        return table[state][:b]

    # Exhaustive: best root-to-leaf path at depth 2.
    best = max(((t1, s1, t2, s2) for t1, s1 in table[""]
                for t2, s2 in table[t1]), key=lambda p: p[1] + p[3])
    out = kamath_tree_of_thoughts("", 2, 2, model, beam=2)
    assert out["best_state"] == best[2]
    assert out["estimate"] == pytest.approx(best[1] + best[3])
    assert out["best_path"] == [best[0], best[2]]
    # Greedy (beam 1) takes the locally better root and misses it.
    greedy = kamath_tree_of_thoughts("", 2, 2, model, beam=1)
    assert greedy["best_state"] == "bb"
    assert greedy["estimate"] < out["estimate"]


def test_kmtot_enforces_the_branch_factor():
    from morie.fn.kmtot import kamath_tree_of_thoughts
    with pytest.raises(ValueError):
        kamath_tree_of_thoughts("", 2, 1,
                                lambda s, b: [("x", 1.0)] * 3)
    with pytest.raises(ValueError):
        kamath_tree_of_thoughts("", 2, 1, lambda s, b: [])


# ── kmtoxg ────────────────────────────────────────────────────────
def test_kmtoxg_accepts_the_three_result_shapes():
    from morie.fn.kmtoxg import kamath_toxigen_score
    assert kamath_toxigen_score("t", lambda x: 0.42)["estimate"] == \
        pytest.approx(0.42)
    assert kamath_toxigen_score("t", lambda x: {"toxic": 0.42})["estimate"] \
        == pytest.approx(0.42)
    assert kamath_toxigen_score("t", lambda x: (0.58, 0.42))["estimate"] == \
        pytest.approx(0.42)
    assert kamath_toxigen_score("t", lambda x: 0.42, threshold=0.4)["toxic"]


def test_kmtoxg_refuses_a_logit():
    from morie.fn.kmtoxg import kamath_toxigen_score
    with pytest.raises(ValueError):
        kamath_toxigen_score("t", lambda x: 3.7)
    with pytest.raises(ValueError):
        kamath_toxigen_score("t", lambda x: (0.6, 0.6))


# ── kmvera ────────────────────────────────────────────────────────
def test_kmvera_equals_the_explicit_diagonal_product():
    from morie.fn.kmvera import kamath_vera_adapter
    d, k, r = 3, 4, 2
    W0 = rand_matrix(d, k, seed=281)
    A = rand_matrix(r, k, seed=283)
    B = rand_matrix(d, r, seed=293)
    lb = rand_matrix(1, d, seed=307)[0]
    ld = rand_matrix(1, r, seed=311)[0]
    x = rand_matrix(1, k, seed=313)[0]
    merged = W0 + np.diag(lb) @ B @ np.diag(ld) @ A
    out = kamath_vera_adapter(W0, A, B, lb, ld, x)
    assert out["h"] == pytest.approx(list(merged @ x))
    assert out["n_trainable"] == d + r
    assert out["n_trainable"] < out["n_trainable_lora_equivalent"]


def test_kmvera_diagonal_lengths_checked():
    from morie.fn.kmvera import kamath_vera_adapter
    with pytest.raises(ValueError):
        kamath_vera_adapter([[1.0]], [[1.0]], [[1.0]], [1.0, 1.0], [1.0],
                            [1.0])


# ── kmverb ────────────────────────────────────────────────────────
def test_kmverb_sums_label_word_probabilities():
    from morie.fn.kmverb import kamath_verbalizer_mapping
    logits = [1.0, -0.5, 2.0, 0.25, -2.0]
    vocab = ["good", "great", "bad", "awful", "meh"]
    vmap = {"pos": ["good", "great"], "neg": ["bad", "awful"]}
    p = softmax(logits)
    out = kamath_verbalizer_mapping(logits, vocab, vmap)
    assert out["probabilities"]["pos"] == pytest.approx(p[0] + p[1])
    assert out["probabilities"]["neg"] == pytest.approx(p[2] + p[3])
    assert out["mass_outside"] == pytest.approx(p[4])
    assert out["prediction"] == "neg"
    assert sum(out["normalized"].values()) == pytest.approx(1.0)


def test_kmverb_overlapping_classes_refused():
    from morie.fn.kmverb import kamath_verbalizer_mapping
    with pytest.raises(ValueError):
        kamath_verbalizer_mapping([0.0, 0.0], ["a", "b"],
                                  {"x": ["a"], "y": ["a", "b"]})
    with pytest.raises(ValueError):
        kamath_verbalizer_mapping([0.0, 0.0], ["a", "b"], {"x": ["z"]})


# ── kmw2v ─────────────────────────────────────────────────────────
def test_kmw2v_matches_a_hand_written_log_softmax():
    from morie.fn.kmw2v import kamath_word2vec_skipgram
    V = rand_matrix(5, 3, seed=317)
    U = rand_matrix(5, 3, seed=331)
    centers = [0, 0, 2, 4]
    contexts = [1, 2, 3, 0]
    want = sum(math.log(softmax(V[c] @ U.T)[o])
               for c, o in zip(centers, contexts))
    out = kamath_word2vec_skipgram(centers, contexts, V, U)
    assert out["log_likelihood"] == pytest.approx(want)
    assert out["estimate"] == pytest.approx(want / 4)
    assert sum(out["probabilities"]) < 4.0        # each is a probability


def test_kmw2v_rejects_self_context_and_bad_indices():
    from morie.fn.kmw2v import kamath_word2vec_skipgram
    V = U = [[1.0], [0.0]]
    with pytest.raises(ValueError):
        kamath_word2vec_skipgram([0], [0], V, U)
    with pytest.raises(ValueError):
        kamath_word2vec_skipgram([0], [5], V, U)


# ── kmyarn ────────────────────────────────────────────────────────
def test_kmyarn_scales_low_frequencies_more_than_high_ones():
    from morie.fn.kmyarn import kamath_yarn_context_extrapolation
    base, s, d = 10000.0, 8.0, 16
    out = kamath_yarn_context_extrapolation(base, s, d)
    half = d // 2
    want = [base ** (-2 * i / d) * s ** (-2 * i / d) for i in range(half)]
    assert out["theta_new"] == pytest.approx(want)
    assert out["scale_factors"][0] == pytest.approx(1.0)      # untouched
    ratio = out["theta_new"][-1] / out["theta"][-1]
    assert ratio == pytest.approx(s ** (-(d - 2) / d))
    assert ratio < 1.0


def test_kmyarn_ramp_blends_and_array_input_checked():
    from morie.fn.kmyarn import kamath_yarn_context_extrapolation
    freqs = [1.0, 0.5, 0.25, 0.125]
    plain = kamath_yarn_context_extrapolation(freqs, 4.0, 8)
    ramped = kamath_yarn_context_extrapolation(freqs, 4.0, 8, ramp=(1, 3))
    assert ramped["theta_new"][0] == pytest.approx(freqs[0])
    assert ramped["theta_new"][3] == pytest.approx(plain["theta_new"][3])
    assert ramped["theta_new"][2] != pytest.approx(plain["theta_new"][2])
    with pytest.raises(ValueError):
        kamath_yarn_context_extrapolation([1.0, 0.5], 2.0, 8)
