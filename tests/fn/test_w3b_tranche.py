# morie.fn -- test file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent-route checks for the w3b Kamath tranche.

Every expected number here is derived by hand (log sums, closed
forms, brute force over LP vertices, finite differences) or by a
composition identity between two modules -- never by re-running the
module under test. A mean-of-inputs stub fails all of them.
"""

import math
from itertools import permutations

from morie.fn import _array_core as np
import pytest

from morie.fn.km110 import kamath_ch7_rrf_score
from morie.fn.km111 import kamath_ch7_faithfulness_metric
from morie.fn.km112 import kamath_ch7_answer_relevance
from morie.fn.km113 import kamath_ch8_perplexity
from morie.fn.km114 import kamath_ch8_bleu_precision
from morie.fn.km115 import kamath_ch8_bleu_n_geom_mean
from morie.fn.km116 import kamath_ch8_brevity_penalty
from morie.fn.km117 import kamath_ch8_bleu_final
from morie.fn.km118 import kamath_ch8_rouge_n
from morie.fn.km119 import kamath_ch8_bertscore_recall
from morie.fn.km120 import kamath_ch8_bertscore_precision
from morie.fn.km121 import kamath_ch8_bertscore_f1
from morie.fn.km122 import kamath_ch8_wmd
from morie.fn.km123 import kamath_ch8_moverscore_distance
from morie.fn.km124 import kamath_ch8_ngram_embedding
from morie.fn.km125 import kamath_ch8_ngram_weight
from morie.fn.km126 import kamath_ch8_smd
from morie.fn.km127 import kamath_ch8_geval_score
from morie.fn.km128 import kamath_ch8_pass_at_k
from morie.fn.km129 import kamath_ch9_modality_encoder
from morie.fn.km130 import kamath_ch9_input_alignment_loss
from morie.fn.km131 import kamath_ch9_input_projector
from morie.fn.km132 import kamath_ch9_llm_signal_tokens
from morie.fn.km133 import kamath_ch9_clip_image_to_text
from morie.fn.km134 import kamath_ch9_clip_text_to_image
from morie.fn.km135 import kamath_ch9_clip_contrastive_total
from morie.fn.km136 import kamath_ch9_mml_vlm_loss
from morie.fn.km137 import kamath_ch9_itm_hard_negative
from morie.fn.km138 import kamath_ch9_simvlm_mlm
from morie.fn.km139 import kamath_ch9_simvlm_prefixlm
from morie.fn.km140 import kamath_ch9_moc_loss
from morie.fn.km141 import kamath_ch9_itm_loss
from morie.fn.km142 import kamath_ch9_itg_loss
from morie.fn.km143 import kamath_ch9_fom_loss
from morie.fn.km144 import kamath_ch9_mm_instr_predict
from morie.fn.km145 import kamath_ch9_mmllm_autoregressive
from morie.fn.km146 import kamath_ch9_output_projector_mse
from morie.fn.km147 import kamath_ch9_output_alignment
from morie.fn.km148 import kamath_ch9_ldm_loss
from morie.fn.km149 import kamath_ch9_flamingo_factorized
from morie.fn.km150 import kamath_ch9_flamingo_dataset_mix
from morie.fn.km3h import kamath_3h_alignment
from morie.fn.kmadal import kamath_adalora_rank_allocation
from morie.fn.kmadap import kamath_houlsby_adapter
from morie.fn.kmalbi import kamath_alibi_bias
from morie.fn.kmap import kamath_autoprompt_gradient_search
from morie.fn.kmarel import kamath_ragas_answer_relevance
from morie.fn.kmbleu import kamath_bleu_score
from morie.fn.kmbm25 import kamath_bm25_score
from morie.fn.kmbon import kamath_best_of_n_sampling
from morie.fn.kmbrad import kamath_bradley_terry_preference
from morie.fn.kmbsco import kamath_bertscore
from morie.fn.kmcai import kamath_constitutional_ai_loop
from morie.fn.kmcap import kamath_expert_capacity_factor
from morie.fn.kmcchr import kamath_christiano_deep_rl_feedback
from morie.fn.kmchin import kamath_chinchilla_compute_optimal
from morie.fn.kmcot import kamath_chain_of_thought
from morie.fn.kmcrag import kamath_corrective_rag
from morie.fn.kmcrb import kamath_cross_encoder_rerank
from morie.fn.kmcrel import kamath_ragas_context_relevance
from morie.fn.kmcrwd import kamath_crowspairs_bias
from morie.fn.kmdbq import kamath_double_quantization
from morie.fn.kmdp import kamath_differential_privacy
from morie.fn.kmdpok import kamath_dpo_loss
from morie.fn.kmdpr import kamath_dense_passage_retrieval
from morie.fn.kmemer import kamath_emergent_abilities
from morie.fn.kmexp import kamath_memorization_exposure
from morie.fn.kmfact import kamath_factscore

TOL = 1e-12


def lcg(n, seed=7):
    """Deterministic uniforms -- the house LCG, no RNG state."""
    s = seed
    out = []
    for _ in range(n):
        s = (1664525 * s + 1013904223) % 2 ** 32
        out.append((s + 0.5) / 2 ** 32)
    return out


# --------------------------------------------------------------- Ch 7

def test_km110_rrf_hand_sum_and_monotone():
    out = kamath_ch7_rrf_score([1, 3])
    assert abs(out["estimate"] - (1 / 61 + 1 / 63)) < TOL
    assert out["scores"][0] > out["scores"][1]
    with pytest.raises(ValueError):
        kamath_ch7_rrf_score([0])


def test_km111_faithfulness_counts():
    assert kamath_ch7_faithfulness_metric([1, 1, 0, 1])["estimate"] == 0.75
    assert kamath_ch7_faithfulness_metric([True, False])["estimate"] == 0.5
    with pytest.raises(ValueError):
        kamath_ch7_faithfulness_metric([])


def test_km112_answer_relevance_cosine():
    # cos(45 deg) = 1/sqrt(2), cos(0) = 1  ->  mean is their average
    out = kamath_ch7_answer_relevance([[1.0, 1.0], [2.0, 0.0]],
                                      [1.0, 0.0])
    assert abs(out["estimate"] - (1 / math.sqrt(2) + 1) / 2) < TOL
    with pytest.raises(ValueError):
        kamath_ch7_answer_relevance([[0.0, 0.0]], [1.0, 0.0])


# --------------------------------------------------------------- Ch 8

def test_km113_perplexity_uniform_equals_vocab_size():
    # a model that is uniform over V has perplexity exactly V
    V = 8
    out = kamath_ch8_perplexity(list("abcd"), p_theta=[1 / V] * 4)
    assert abs(out["estimate"] - V) < 1e-9
    hand = math.exp(-(math.log(0.5) + math.log(0.1)) / 2)
    assert abs(kamath_ch8_perplexity(["a", "b"],
                                     p_theta=[0.5, 0.1])["estimate"]
               - hand) < 1e-12


def test_km113_callable_scorer_and_length_check():
    out = kamath_ch8_perplexity(["a", "b"],
                                p_theta=lambda x, pre: 0.5 ** (len(pre) + 1))
    # p = 0.5 then 0.25 -> mean nll = (log2 + log4)/2 = 1.5 log 2
    assert abs(out["estimate"] - math.exp(1.5 * math.log(2))) < 1e-12
    with pytest.raises(ValueError):
        kamath_ch8_perplexity(["a"], N=3, p_theta=[0.5])


def test_km114_precision_ratio_and_clipping_guard():
    out = kamath_ch8_bleu_precision([[3, 4], [1, 3]])
    assert out["p_n"] == [0.75, 1 / 3]
    with pytest.raises(ValueError):
        kamath_ch8_bleu_precision([[5, 4]])
    with pytest.raises(ValueError):
        kamath_ch8_bleu_precision([[0, 0]])


def test_km115_geometric_mean_matches_nth_root():
    p = [0.2, 0.4, 0.8]
    hand = (0.2 * 0.4 * 0.8) ** (1 / 3)
    assert abs(kamath_ch8_bleu_n_geom_mean(p)["estimate"] - hand) < 1e-12
    assert kamath_ch8_bleu_n_geom_mean([0.5, 0.0])["estimate"] == 0.0


def test_km116_brevity_penalty_branches():
    assert kamath_ch8_brevity_penalty(7, 5)["estimate"] == 1.0
    assert abs(kamath_ch8_brevity_penalty(5, 5)["estimate"] - 1.0) < TOL
    assert abs(kamath_ch8_brevity_penalty(4, 8)["estimate"]
               - math.exp(-1.0)) < TOL
    with pytest.raises(ValueError):
        kamath_ch8_brevity_penalty(0, 5)


def test_km117_bleu_is_bp_times_geometric_mean():
    p = [0.2, 0.4, 0.8]
    bp = kamath_ch8_brevity_penalty(4, 8)["estimate"]
    gm = kamath_ch8_bleu_n_geom_mean(p)["estimate"]
    assert abs(kamath_ch8_bleu_final(bp, p)["estimate"] - bp * gm) < TOL


def test_km118_rouge_hand_count_and_perfect_recall():
    refs = [["a", "b", "c"], ["a", "d"]]
    out = kamath_ch8_rouge_n(refs, 1, candidate=["a", "b"])
    # matches: a,b from ref1 (2) + a from ref2 (1) = 3 of 5 ref unigrams
    assert abs(out["estimate"] - 3 / 5) < TOL
    same = kamath_ch8_rouge_n([["a", "b"]], 2, candidate=["a", "b"])
    assert same["estimate"] == 1.0


def test_km118_clipping_stops_double_counting():
    out = kamath_ch8_rouge_n([["a", "a"]], 1, candidate=["a"])
    assert out["estimate"] == 0.5      # 1 of the 2 reference "a"s


def test_km119_120_recall_precision_are_transposes():
    x = [[1.0, 0.0], [0.0, 1.0]]
    xh = [[1.0, 0.0]]
    assert kamath_ch8_bertscore_recall(x, xh)["estimate"] == 0.5
    assert kamath_ch8_bertscore_precision(x, xh)["estimate"] == 1.0
    # precision(x, xhat) IS recall(xhat, x)
    a = kamath_ch8_bertscore_precision(x, xh)["estimate"]
    b = kamath_ch8_bertscore_recall(xh, x)["estimate"]
    assert a == b


def test_km119_normalize_gives_cosines_bounded_by_one():
    x = [[3.0, 4.0], [0.0, 2.0]]
    out = kamath_ch8_bertscore_recall(x, [[6.0, 8.0]], normalize=True)
    assert abs(out["per_token"][0] - 1.0) < 1e-12   # parallel vectors
    assert all(v <= 1.0 + 1e-12 for v in out["per_token"])


def test_km121_f1_bounds_and_equality():
    f = kamath_ch8_bertscore_f1(0.6, 0.6)["estimate"]
    assert abs(f - 0.6) < TOL                      # harmonic of equals
    g = kamath_ch8_bertscore_f1(1.0, 0.2)["estimate"]
    assert 0.2 <= g <= (1.0 + 0.2) / 2             # min <= H <= mean
    with pytest.raises(ValueError):
        kamath_ch8_bertscore_f1(0.0, 0.0)


def test_km122_matches_brute_force_over_permutations():
    # uniform marginals: the optimum of a balanced n x n transport
    # problem is attained at a permutation vertex (Birkhoff), so the
    # LP answer must equal the best permutation cost.
    n = 4
    vals = lcg(n * n, seed=11)
    C = np.array(vals).reshape(n, n) * 10
    a = b = [1.0 / n] * n
    best = min(sum(C[i, p[i]] for i in range(n)) / n
               for p in permutations(range(n)))
    out = kamath_ch8_wmd(a, b, C)
    assert abs(out["estimate"] - best) < 1e-9
    F = np.array(out["flow"])
    assert np.allclose(F.sum(axis=1), a)
    assert np.allclose(F.sum(axis=0), b)
    assert out["optimal"] is True


def test_km122_unbalanced_marginals_and_supplied_plan():
    with pytest.raises(ValueError):
        kamath_ch8_wmd([1.0], [0.5], [[1.0]])
    out = kamath_ch8_wmd([0.5, 0.5], [0.5, 0.5], [[1.0, 3.0], [4.0, 2.0]],
                         F=[[0.0, 0.5], [0.5, 0.0]])
    assert abs(out["estimate"] - (0.5 * 3 + 0.5 * 4)) < TOL
    assert out["optimal"] is False
    with pytest.raises(ValueError):
        kamath_ch8_wmd([0.5, 0.5], [0.5, 0.5], [[1.0, 3.0], [4.0, 2.0]],
                       F=[[0.5, 0.5], [0.0, 0.0]])


def test_km123_euclidean_and_callable_embedding():
    assert kamath_ch8_moverscore_distance([0.0, 0.0],
                                          [3.0, 4.0])["estimate"] == 5.0
    out = kamath_ch8_moverscore_distance("aa", "b",
                                         E=lambda s: [len(s), 0.0])
    assert out["estimate"] == 1.0
    with pytest.raises(ValueError):
        kamath_ch8_moverscore_distance([1.0], [1.0, 2.0])


def test_km124_window_sum_and_bounds():
    assert kamath_ch8_ngram_embedding([1.0, 2.0, 3.0], 1, 2)["estimate"] == 5.0
    vec = kamath_ch8_ngram_embedding([[1.0, 0.0], [0.0, 2.0]], 0, 2)
    assert vec["estimate"] == [1.0, 2.0]
    with pytest.raises(ValueError):
        kamath_ch8_ngram_embedding([1.0, 2.0], 1, 3)


def test_km125_weights_are_a_distribution():
    out = kamath_ch8_ngram_weight([[1.0, 2.0], [3.0, 4.0]])
    assert abs(sum(out["weights"]) - 1.0) < TOL
    assert abs(out["weights"][0] - 0.3) < TOL
    fixed = kamath_ch8_ngram_weight([2.0, 2.0], Z=8.0)
    assert fixed["estimate"] == 0.5
    with pytest.raises(ValueError):
        kamath_ch8_ngram_weight([1.0], Z=0.0)


def test_km126_smd_equals_km123():
    x, y = [1.0, 2.0], [4.0, 6.0]
    assert (kamath_ch8_smd(x, y)["estimate"]
            == kamath_ch8_moverscore_distance(x, y)["estimate"] == 5.0)


def test_km127_geval_expectation_within_score_range():
    out = kamath_ch8_geval_score([1, 2, 3], [0.2, 0.3, 0.5])
    assert abs(out["estimate"] - 2.3) < TOL
    assert 1 <= out["estimate"] <= 3
    with pytest.raises(ValueError):
        kamath_ch8_geval_score([1, 2], [0.5, 0.6])


def test_km128_pass_at_k_edges_and_k_one():
    assert kamath_ch8_pass_at_k(10, 0, 3)["estimate"] == 0.0
    assert kamath_ch8_pass_at_k(10, 10, 3)["estimate"] == 1.0
    # k = 1 is just the empirical pass rate
    assert abs(kamath_ch8_pass_at_k(10, 4, 1)["estimate"] - 0.4) < TOL
    # hand: 1 - C(6,2)/C(10,2) = 1 - 15/45
    assert abs(kamath_ch8_pass_at_k(10, 4, 2)["estimate"]
               - (1 - 15 / 45)) < 1e-12


def test_km128_monotone_in_k():
    vals = [kamath_ch8_pass_at_k(20, 5, k)["estimate"] for k in (1, 3, 9)]
    assert vals[0] < vals[1] < vals[2] <= 1.0
    with pytest.raises(ValueError):
        kamath_ch8_pass_at_k(5, 6, 2)


# --------------------------------------------------------------- Ch 9

def test_km129_encoder_contract():
    out = kamath_ch9_modality_encoder([1.0], lambda z: [0.0, 5.0])
    assert out["estimate"] == 5.0
    with pytest.raises(ValueError):
        kamath_ch9_modality_encoder([1.0], "not callable")
    with pytest.raises(ValueError):
        kamath_ch9_modality_encoder([1.0], lambda z: [float("inf")])


def test_km130_argmin_picks_the_best_candidate():
    mse = lambda y, t: float(np.mean((np.asarray(y) - np.asarray(t)) ** 2))
    add = lambda p, f: np.asarray(p) + np.asarray(f)
    out = kamath_ch9_input_alignment_loss(
        [[[5.0]], [[0.0]], [[2.0]]], [[1.0]], [[1.0]],
        llm=add, loss_fn=mse)
    assert out["argmin"] == 1
    assert out["losses"] == [25.0, 0.0, 4.0]
    with pytest.raises(ValueError):
        kamath_ch9_input_alignment_loss([[0.0]], [[1.0]], [[1.0]],
                                        llm=add)


def test_km131_linear_projector_matches_matmul():
    F = [[1.0, 2.0], [0.0, 1.0]]
    W = [[1.0, 0.0], [3.0, 1.0]]
    out = kamath_ch9_input_projector(F, W)
    assert out["prompts"] == [[7.0, 2.0], [3.0, 1.0]]
    assert abs(out["estimate"]
               - math.sqrt(49 + 4 + 9 + 1)) < 1e-12
    with pytest.raises(ValueError):
        kamath_ch9_input_projector(F, [[1.0]])


def test_km132_signal_token_contract():
    out = kamath_ch9_llm_signal_tokens([[0.0]], [[1.0]],
                                       llm=lambda p, f: ("t", ["a", "b"]))
    assert out["estimate"] == 2 and out["generates_modality"] is True
    with pytest.raises(ValueError):
        kamath_ch9_llm_signal_tokens([[0.0]], [[1.0]],
                                     llm=lambda p, f: "just text")


def test_km133_clip_loss_hand_logsumexp():
    V = [[1.0, 0.0], [0.0, 1.0]]
    L = [[2.0, 0.0], [0.0, 2.0]]
    out = kamath_ch9_clip_image_to_text(V, L, 1.0)
    # each row's logits are (2, 0); loss = log(e^2 + 1) - 2
    hand = math.log(math.exp(2.0) + 1.0) - 2.0
    assert abs(out["estimate"] - hand) < 1e-12
    assert out["estimate"] > 0


def test_km133_lower_temperature_sharpens():
    V = [[1.0, 0.0], [0.0, 1.0]]
    L = [[1.0, 0.0], [0.0, 1.0]]
    hot = kamath_ch9_clip_image_to_text(V, L, 4.0)["estimate"]
    cold = kamath_ch9_clip_image_to_text(V, L, 0.25)["estimate"]
    assert cold < hot
    with pytest.raises(ValueError):
        kamath_ch9_clip_image_to_text(V, L, 0.0)


def test_km134_is_km133_with_swapped_modalities():
    A = [[1.0, 2.0], [0.0, 1.0]]
    B = [[1.0, 0.0], [3.0, 1.0]]
    assert (kamath_ch9_clip_text_to_image(A, B, 0.7)["estimate"]
            == kamath_ch9_clip_image_to_text(A, B, 0.7)["estimate"])


def test_km135_total_sums_and_rejects_negatives():
    assert kamath_ch9_clip_contrastive_total(0.25, 0.75)["estimate"] == 1.0
    with pytest.raises(ValueError):
        kamath_ch9_clip_contrastive_total(-0.1, 0.5)


def test_km136_mml_is_a_sum_of_negative_logs():
    out = kamath_ch9_mml_vlm_loss([0.5, 0.25], [0.5])
    hand = math.log(2) + math.log(4) + math.log(2)
    assert abs(out["estimate"] - hand) < 1e-12
    assert abs(out["positive_loss"] - (math.log(2) + math.log(4))) < 1e-12
    with pytest.raises(ValueError):
        kamath_ch9_mml_vlm_loss([1.5], [0.5])


def test_km137_hard_negatives_delegate_to_km136():
    a = kamath_ch9_itm_hard_negative([0.6], [0.7])["estimate"]
    b = kamath_ch9_mml_vlm_loss([0.6], [0.7])["estimate"]
    assert a == b


def test_km138_mlm_mean_and_visual_contract():
    out = kamath_ch9_simvlm_mlm(None, [0.5, 1.0, 0.25], [[0.0]], [0, 2])
    assert abs(out["estimate"]
               - (math.log(2) + math.log(4)) / 2) < 1e-12
    assert out["n_image_regions"] == 1
    with pytest.raises(ValueError):
        kamath_ch9_simvlm_mlm(None, [0.5], None, [0])


def test_km139_prefixlm_sums_the_suffix():
    out = kamath_ch9_simvlm_prefixlm(None, [0.5, 0.5, 0.25], 1)
    assert abs(out["estimate"] - math.log(8)) < 1e-12
    batch = kamath_ch9_simvlm_prefixlm(None, [[0.5, 0.5], [0.5, 0.25]], 1)
    assert abs(batch["estimate"]
               - (math.log(2) + math.log(4)) / 2) < 1e-12
    with pytest.raises(ValueError):
        kamath_ch9_simvlm_prefixlm(None, [0.5, 0.5], 2)


def test_km140_moc_index_and_onehot_labels_agree():
    G = [[0.5, 0.5], [0.25, 0.75]]
    a = kamath_ch9_moc_loss(None, None, [[0.0]], G, labels=[0, 1])
    b = kamath_ch9_moc_loss(None, None, [[0.0]], G,
                            labels=[[1, 0], [0, 1]])
    assert abs(a["estimate"] - b["estimate"]) < TOL
    assert abs(a["estimate"] - (math.log(2) - math.log(0.75))) < 1e-12
    assert a["as_printed"] == -a["estimate"]
    with pytest.raises(ValueError):
        kamath_ch9_moc_loss(None, None, None, [[0.5, 0.6]], labels=[0])


def test_km141_itm_binary_cross_entropy():
    out = kamath_ch9_itm_loss([0.9, 0.2], None, None, [1, 0])
    hand = (-math.log(0.9) - math.log(0.8)) / 2
    assert abs(out["estimate"] - hand) < 1e-12
    perfect = kamath_ch9_itm_loss([1.0, 0.0], None, None, [1, 0])
    assert perfect["estimate"] == 0.0
    with pytest.raises(ValueError):
        kamath_ch9_itm_loss([0.5], None, None, [2])


def test_km142_is_the_sum_of_per_pair_km145():
    seqs = [[0.5, 0.5], [0.25]]
    total = kamath_ch9_itg_loss(None, seqs)["estimate"]
    parts = sum(kamath_ch9_mmllm_autoregressive(s, None)["estimate"]
                for s in seqs)
    assert abs(total - parts) < TOL
    assert abs(total - 2 * math.log(4)) < 1e-12
    with pytest.raises(ValueError):
        kamath_ch9_itg_loss(["one context"], seqs)


def test_km143_fom_picks_the_labelled_cells():
    P = [[0.5, 0.5], [0.25, 0.75]]
    out = kamath_ch9_fom_loss([0, 1], [1, 0], P=P)
    assert abs(out["estimate"] - (math.log(2) + math.log(4))) < 1e-12
    with pytest.raises(ValueError):
        kamath_ch9_fom_loss([0], [0], R=2, P=P)
    with pytest.raises(ValueError):
        kamath_ch9_fom_loss([0], [0])


def test_km144_prediction_contract():
    out = kamath_ch9_mm_instr_predict("q", "img", lambda i, m: i + m)
    assert out["answer"] == "qimg"
    out2 = kamath_ch9_mm_instr_predict("q", "img", 3.0,
                                       f=lambda i, m, th: th * 2)
    assert out2["answer"] == 6.0
    with pytest.raises(ValueError):
        kamath_ch9_mm_instr_predict("q", "img", 3.0)


def test_km145_sequence_nll_and_probability():
    out = kamath_ch9_mmllm_autoregressive([0.5, 0.25, 0.5], None)
    assert abs(out["estimate"] - math.log(16)) < 1e-12
    assert abs(out["sequence_probability"] - 0.0625) < TOL
    assert abs(out["mean_nll"] - out["estimate"] / 3) < TOL
    cb = kamath_ch9_mmllm_autoregressive("ignored", "img",
                                         theta=lambda r, i: [0.5])
    assert abs(cb["estimate"] - math.log(2)) < 1e-12


def test_km146_mse_and_candidate_argmin():
    out = kamath_ch9_output_projector_mse([[1.0, 2.0]],
                                          lambda t: [[1.0, 0.0]], None)
    assert out["estimate"] == 2.0
    stack = kamath_ch9_output_projector_mse(
        [[[1.0, 2.0]], [[1.0, 0.0]]], [[1.0, 0.0]], None)
    assert stack["argmin"] == 1 and stack["estimate"] == 0.0
    with pytest.raises(ValueError):
        kamath_ch9_output_projector_mse([[1.0]], [[1.0, 0.0]], None)


def test_km147_output_projector_matmul():
    out = kamath_ch9_output_alignment([[1.0, 2.0]], [[0.0], [2.0]])
    assert out["features"] == [[4.0]]
    cb = kamath_ch9_output_alignment([[1.0]], lambda s: s * 3)
    assert cb["features"] == [[3.0]]
    with pytest.raises(ValueError):
        kamath_ch9_output_alignment([[1.0, 2.0]])


def test_km148_squared_l2_noise_error():
    out = kamath_ch9_ldm_loss([[3.0, 4.0], [0.0, 0.0]],
                              [[0.0, 0.0], [0.0, 0.0]], [[0.0]],
                              eps_net=lambda z, t, h: np.zeros((2, 2)))
    assert out["per_sample"] == [25.0, 0.0]
    assert out["estimate"] == 12.5
    with pytest.raises(ValueError):
        kamath_ch9_ldm_loss([[1.0]], [[0.0]], [[0.0]])


def test_km149_product_and_log_agree():
    out = kamath_ch9_flamingo_factorized([0.5, 0.25, 0.5])
    assert abs(out["estimate"] - 0.0625) < TOL
    assert abs(out["log_prob"] - math.log(0.0625)) < 1e-12
    assert abs(out["nll"] + out["log_prob"]) < TOL
    with pytest.raises(ValueError):
        kamath_ch9_flamingo_factorized([0.5], L=2)


def test_km150_weighted_mix_matches_hand_sum():
    D = [[[0.5], [0.25]], [[0.5, 0.5]]]
    out = kamath_ch9_flamingo_dataset_mix(D, [0.5, 2.0])
    d0 = (math.log(2) + math.log(4)) / 2
    d1 = math.log(4)
    assert abs(out["estimate"] - (0.5 * d0 + 2.0 * d1)) < 1e-12
    assert abs(out["per_dataset_nll"][0] - d0) < 1e-12
    with pytest.raises(ValueError):
        kamath_ch9_flamingo_dataset_mix(D, [1.0])


# ----------------------------------------------------- Ch 1-7 concepts

def test_km3h_weighted_and_default_weights():
    out = kamath_3h_alignment(0.8, 0.6, 1.0, [0.5, 0.3, 0.2])
    assert abs(out["estimate"] - 0.78) < TOL
    even = kamath_3h_alignment(0.9, 0.3, 0.6)
    assert abs(even["estimate"] - 0.6) < 1e-12
    with pytest.raises(ValueError):
        kamath_3h_alignment(0.5, 0.5, 0.5, [0.5, 0.5])


def test_kmadal_prunes_to_exactly_target_rank():
    P = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    s = [5.0, 0.1, 3.0]
    Q = P
    out = kamath_adalora_rank_allocation(P, s, Q, target_rank=2)
    dW = np.array(out["Delta_W"])
    assert np.linalg.matrix_rank(dW) == 2
    assert out["kept"] == [0, 2]
    assert dW[1, 1] == 0.0 and dW[0, 0] == 5.0
    full = kamath_adalora_rank_allocation(P, s, Q)
    assert np.allclose(np.array(full["Delta_W"]), np.diag(s))


def test_kmadal_importance_overrides_magnitude():
    P = [[1.0, 0.0], [0.0, 1.0]]
    out = kamath_adalora_rank_allocation(P, [5.0, 1.0], P,
                                         importance=[0.1, 9.0],
                                         target_rank=1)
    assert out["kept"] == [1]
    with pytest.raises(ValueError):
        kamath_adalora_rank_allocation(P, [1.0, 2.0], P, target_rank=3)


def test_kmadap_residual_identity_and_bottleneck_shape():
    # GELU(0) = 0, so a zero pre-activation leaves h untouched
    out = kamath_houlsby_adapter([1.0, 2.0, 3.0], [[0.0, 0.0, 0.0]],
                                 [[0.0], [0.0], [0.0]])
    assert out["h_adapted"] == [[1.0, 2.0, 3.0]]
    assert out["estimate"] == 0.0
    with pytest.raises(ValueError):
        kamath_houlsby_adapter([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]],
                               [[1.0, 0.0], [0.0, 1.0]])


def test_kmadap_tanh_approximation_tracks_the_exact_gelu():
    exact = kamath_houlsby_adapter([1.0, 0.0], [[1.0, 0.0]],
                                   [[1.0], [0.0]])["h_adapted"][0][0]
    approx = kamath_houlsby_adapter([1.0, 0.0], [[1.0, 0.0]],
                                    [[1.0], [0.0]],
                                    approximate="tanh")["h_adapted"][0][0]
    assert abs(exact - approx) < 1e-3
    with pytest.raises(ValueError):
        kamath_houlsby_adapter([1.0, 0.0], [[1.0, 0.0]], [[1.0], [0.0]],
                               approximate="relu")


def test_kmadap_exact_gelu_number():
    out = kamath_houlsby_adapter([1.0, 0.0], [[1.0, 0.0]],
                                 [[1.0], [0.0]])
    g = 0.5 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))
    assert abs(out["h_adapted"][0][0] - (1.0 + g)) < 1e-12
    assert abs(out["bottleneck"][0][0] - g) < 1e-12


def test_kmalbi_zero_slope_is_plain_attention():
    from morie.fn.attsdp import scaled_dot_product_attention
    Q = [[1.0, 0.0], [0.0, 1.0]]
    K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[1.0], [0.0]]
    a = kamath_alibi_bias(Q, K, V, 0.0)["output"][0]
    b = scaled_dot_product_attention(Q, K, V)["output"]
    assert np.allclose(np.array(a), np.array(b))


def test_kmalbi_slope_penalizes_distance():
    Q = [[1.0, 0.0], [1.0, 0.0]]
    K = [[1.0, 0.0], [1.0, 0.0]]     # identical keys: only the bias acts
    V = [[1.0], [0.0]]
    A = np.array(kamath_alibi_bias(Q, K, V, 1.0)["attention"][0])
    assert abs(A.sum(axis=1) - 1).max() < 1e-12
    # row 1 attends less to the distant token 0 than to itself
    assert A[1, 0] < A[1, 1]
    # exact: weights proportional to exp(j - i)
    assert abs(A[1, 0] / A[1, 1] - math.exp(-1.0)) < 1e-12
    with pytest.raises(ValueError):
        kamath_alibi_bias(Q, K, V, -0.5)


def test_kmap_exact_search_picks_the_lowest_loss():
    losses = {"a": 1.0, "b": 0.5, "c": 0.75}
    out = kamath_autoprompt_gradient_search(
        ["x", None], [1], lambda tpl, d: losses[tpl[1]],
        vocab=["a", "b", "c"])
    assert out["trigger_tokens"] == ["b"] and out["estimate"] == 0.5


def test_kmap_gradient_route_takes_the_argmax_score():
    out = kamath_autoprompt_gradient_search(
        [None, "y"], [1], lambda tpl, d: 0.0, vocab=["a", "b", "c"],
        grad_fn=lambda tpl, d, i: [0.1, 0.2, 9.0])
    assert out["trigger_tokens"] == ["c"]
    with pytest.raises(ValueError):
        kamath_autoprompt_gradient_search(["x", "y"], [1],
                                          lambda t, d: 0.0, vocab=["a"])


def test_kmarel_matches_km112_on_the_same_embeddings():
    E_g = [[1.0, 1.0], [2.0, 0.0]]
    out = kamath_ragas_answer_relevance("ans", [1.0, 0.0],
                                        lambda a: E_g)
    ref = kamath_ch7_answer_relevance(E_g, [1.0, 0.0])["estimate"]
    assert out["estimate"] == ref
    with pytest.raises(ValueError):
        kamath_ragas_answer_relevance("ans", [1.0, 0.0], lambda a: [])


def test_kmbleu_identical_text_scores_one():
    toks = ["the", "cat", "sat", "on", "the", "mat"]
    out = kamath_bleu_score(toks, [toks], max_n=4)
    assert abs(out["bleu"] - 1.0) < 1e-12
    assert out["p_n"] == [1.0, 1.0, 1.0, 1.0]


def test_kmbleu_brevity_and_clipping_by_hand():
    out = kamath_bleu_score(["the", "the"], [["the", "cat"]], max_n=1)
    # clipped: "the" appears once in the reference -> 1 of 2 unigrams
    assert out["p_n"] == [0.5]
    assert out["brevity_penalty"] == 1.0        # c == r -> exp(0)
    assert abs(out["bleu"] - 0.5) < 1e-12
    short = kamath_bleu_score(["the"], [["the", "cat", "sat"]], max_n=1)
    assert abs(short["bleu"] - math.exp(1 - 3.0)) < 1e-12


def test_kmbm25_hand_value_and_saturation():
    idf = {"a": 1.0}
    one = kamath_bm25_score(["a"], ["a", "b"], idf, 2.0)["estimate"]
    assert abs(one - 1.0) < 1e-12
    two = kamath_bm25_score(["a"], ["a", "a"], idf, 2.0)["estimate"]
    # tf 2 with |d| = 2: 2*2.5 / (2 + 1.5) = 10/7
    assert abs(two - 10 / 7) < 1e-12
    assert two > one                       # saturating but increasing
    absent = kamath_bm25_score(["z"], ["a"], {"z": 3.0}, 1.0)
    assert absent["estimate"] == 0.0


def test_kmbm25_b_zero_removes_length_normalization():
    idf = {"a": 1.0}
    short = kamath_bm25_score(["a"], ["a"], idf, 5.0, b=0.0)["estimate"]
    long_ = kamath_bm25_score(["a"], ["a"] + ["z"] * 9, idf, 5.0,
                              b=0.0)["estimate"]
    assert abs(short - long_) < 1e-12
    with pytest.raises(ValueError):
        kamath_bm25_score(["a"], ["a"], idf, 0.0)


def test_kmbon_argmax_and_reward_fn():
    out = kamath_best_of_n_sampling(["a", "b", "c"], [0.1, 0.9, 0.4])
    assert (out["best"], out["best_index"]) == ("b", 1)
    assert abs(out["reward_spread"] - 0.8) < TOL
    fn = kamath_best_of_n_sampling(["aa", "b"], reward_fn=lambda x, y: len(y))
    assert fn["best"] == "aa"
    with pytest.raises(ValueError):
        kamath_best_of_n_sampling([], [])


def test_kmbrad_sigmoid_and_antisymmetry():
    p = kamath_bradley_terry_preference(2.0, 0.0)["estimate"]
    assert abs(p - 1 / (1 + math.exp(-2.0))) < 1e-15
    q = kamath_bradley_terry_preference(0.0, 2.0)["estimate"]
    assert abs(p + q - 1.0) < 1e-15
    big = kamath_bradley_terry_preference(800.0, 0.0)["estimate"]
    assert big == 1.0                       # stable, not overflow-nan


def test_kmbsco_hand_scores_and_perfect_match():
    emb = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    out = kamath_bertscore(["a"], ["a", "b"], emb)
    assert (out["precision"], out["recall"]) == (1.0, 0.5)
    assert abs(out["f1"] - 2 / 3) < 1e-12
    same = kamath_bertscore(["a", "b"], ["a", "b"], emb)
    assert abs(same["f1"] - 1.0) < 1e-12


def test_kmbsco_callable_encoder():
    out = kamath_bertscore(["x"], ["x"],
                           lambda toks: [[3.0, 4.0] for _ in toks])
    assert abs(out["f1"] - 1.0) < 1e-12     # cosine of parallel vectors
    with pytest.raises(ValueError):
        kamath_bertscore([], ["x"], lambda t: [[1.0]])


def test_kmcai_revisions_compose_in_order():
    def m(stage, principle, response, critique):
        return "crit" if stage == "critique" else response + principle
    out = kamath_constitutional_ai_loop("y", ["1", "2"], m)
    assert out["revised_response"] == "y12"
    assert out["history"][1]["response_before"] == "y1"
    with pytest.raises(ValueError):
        kamath_constitutional_ai_loop("y", [], m)


def test_kmcap_capacity_and_slots():
    out = kamath_expert_capacity_factor(100, 4, 1.25)
    assert (out["capacity"], out["slots"]) == (31.25, 32)
    tight = kamath_expert_capacity_factor(100, 4, 0.5)
    assert tight["capacity"] == 12.5 and tight["min_dropped"] == 48
    with pytest.raises(ValueError):
        kamath_expert_capacity_factor(100, 0, 1.0)


def test_kmcchr_sum_matches_per_pair_bradley_terry():
    pairs = [(2.0, 0.0), (0.0, 1.0)]
    out = kamath_christiano_deep_rl_feedback(pairs, lambda s: s)
    hand = (math.log1p(math.exp(-2.0)) + math.log1p(math.exp(1.0)))
    assert abs(out["estimate"] - hand) < 1e-12
    assert abs(out["mean_loss"] - hand / 2) < 1e-12
    assert out["pair_accuracy"] == 0.5
    with pytest.raises(ValueError):
        kamath_christiano_deep_rl_feedback(pairs, 3.0)


def test_kmchin_split_satisfies_the_compute_identity():
    out = kamath_chinchilla_compute_optimal(1.2e10)
    assert abs(out["N_opt"] - 1e4) < 1e-6
    assert abs(out["D_opt"] - 2e5) < 1e-3
    assert abs(out["compute_check"] - 1.2e10) < 1.0
    assert abs(out["D_opt"] / out["N_opt"] - 20.0) < 1e-9
    with pytest.raises(ValueError):
        kamath_chinchilla_compute_optimal(1e10, alpha=0.6, beta=0.6)


def test_kmcot_parses_and_refuses_unmarked_output():
    out = kamath_chain_of_thought("2+2?", lambda p: "add. Answer: 4")
    assert (out["answer"], out["reasoning"]) == ("4", "add.")
    with pytest.raises(ValueError):
        kamath_chain_of_thought("q", lambda p: "no marker here")
    custom = kamath_chain_of_thought("q", lambda p: "a|b",
                                     parser=lambda t: tuple(t.split("|")))
    assert custom["answer"] == "b"


def test_kmcrag_three_routes():
    docs = ["d1", "d2"]
    hi = kamath_corrective_rag("q", docs,
                               lambda q, d: 0.9 if d == "d1" else 0.1,
                               0.8, 0.2)
    assert (hi["action"], hi["ctx"]) == ("use_docs", ["d1"])
    lo = kamath_corrective_rag("q", docs, lambda q, d: 0.05, 0.8, 0.2)
    assert (lo["action"], lo["ctx"]) == ("fallback_web", [])
    mid = kamath_corrective_rag("q", docs, lambda q, d: 0.5, 0.8, 0.2)
    assert mid["action"] == "mixed" and mid["ctx"] == docs
    with pytest.raises(ValueError):
        kamath_corrective_rag("q", docs, lambda q, d: 0.5, 0.2, 0.8)


def test_kmcrb_rerank_orders_and_is_stable():
    out = kamath_cross_encoder_rerank("q", ["aa", "b", "cc"],
                                      lambda q, d: len(d))
    assert out["ranking"] == [0, 2, 1]      # tie 'aa','cc' keeps order
    assert out["reranked"] == ["aa", "cc", "b"]
    top = kamath_cross_encoder_rerank("q", ["aa", "b"],
                                      lambda q, d: len(d), top_k=1)
    assert top["reranked"] == ["aa"]


def test_kmcrel_fraction_and_alignment_check():
    out = kamath_ragas_context_relevance(["s1", "s2", "s3", "s4"],
                                         [1, 0, 1, 1])
    assert out["estimate"] == 0.75 and out["n_relevant"] == 3
    with pytest.raises(ValueError):
        kamath_ragas_context_relevance(["s1"], [1, 0])


def test_kmcrwd_ties_and_extremes():
    assert kamath_crowspairs_bias([-1.0, -3.0],
                                  [-2.0, -2.0])["estimate"] == 0.5
    tie = kamath_crowspairs_bias([-1.0, -1.0], [-1.0, -1.0])
    assert tie["estimate"] == 0.0 and tie["n_ties"] == 2
    biased = kamath_crowspairs_bias([-0.5, -0.5], [-2.0, -3.0])
    assert biased["estimate"] == 1.0 and biased["bias_gap"] == 0.5
    with pytest.raises(ValueError):
        kamath_crowspairs_bias([0.5], [-1.0])


def test_kmdbq_codes_and_error_bound():
    out = kamath_double_quantization([1.0, 0.5, 0.25])
    assert out["scales_int8"][0] == 127
    assert abs(out["shared_const"] - 1.0 / 127) < 1e-15
    # every dequantized scale is within half a step of the original
    for got, want in zip(out["dequantized"], [1.0, 0.5, 0.25]):
        assert abs(got - want) <= out["shared_const"] / 2 + 1e-15
    with pytest.raises(ValueError):
        kamath_double_quantization([0.0, 0.0])


def test_kmdp_detects_a_violation():
    ok = kamath_differential_privacy(1.0, 0.0, [0.5], [0.2])
    assert ok["guarantee"] is True
    assert abs(ok["estimate"] - (math.e * 0.2 - 0.5)) < 1e-12
    bad = kamath_differential_privacy(0.1, 0.0, [0.9], [0.1])
    assert bad["guarantee"] is False and bad["estimate"] < 0
    bare = kamath_differential_privacy(2.0, 0.01)
    assert abs(bare["multiplicative_bound"] - math.exp(2.0)) < 1e-12
    with pytest.raises(ValueError):
        kamath_differential_privacy(-1.0, 0.0)


def test_kmdpok_equals_bradley_terry_on_implicit_rewards():
    from morie.fn.alrmt import alammar_reward_model_training_bt
    out = kamath_dpo_loss([-1.0, -2.0], [-3.0, -1.0],
                          [-2.0, -2.5], [-3.0, -3.0], 2.0)
    rw = [2.0 * (-1.0 + 2.0), 2.0 * (-2.0 + 2.5)]
    rl = [2.0 * (-3.0 + 3.0), 2.0 * (-1.0 + 3.0)]
    ref = alammar_reward_model_training_bt(rw, rl)["estimate"]
    assert abs(out["estimate"] - ref) < TOL
    assert out["implicit_reward_w"] == rw


def test_kmdpok_no_margin_gives_log_two():
    out = kamath_dpo_loss(-1.0, -1.0, -2.0, -2.0, 1.0)
    assert abs(out["estimate"] - math.log(2)) < 1e-12
    with pytest.raises(ValueError):
        kamath_dpo_loss(0.5, -1.0, -1.0, -1.0, 1.0)
    with pytest.raises(ValueError):
        kamath_dpo_loss(-1.0, -1.0, -1.0, -1.0, 0.0)


def test_kmdpr_topk_by_dot_product():
    out = kamath_dense_passage_retrieval(
        [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]], 2)
    assert out["top_k_indices"] == [2, 0]
    assert out["top_k_scores"] == [2.0, 1.0]
    assert out["scores"][1] == 0.0
    with pytest.raises(ValueError):
        kamath_dense_passage_retrieval([1.0, 0.0], [[1.0, 0.0]], 2)


def test_kmemer_jump_and_gate():
    out = kamath_emergent_abilities([1.0, 10.0, 100.0],
                                    [0.1, 0.1, 0.9], 50.0)
    assert abs(out["estimate"] - 0.8) < 1e-12
    assert out["emergent_score"] == [0.0, 0.0, 0.9]
    with pytest.raises(ValueError):
        kamath_emergent_abilities([1.0, 2.0], [0.1, 0.2], 100.0)


def test_kmexp_rank_and_maximum_exposure():
    out = kamath_memorization_exposure(-1.0, [-2.0, -3.0, -0.5])
    assert out["rank"] == 2 and abs(out["estimate"] - 1.0) < TOL
    best = kamath_memorization_exposure(-0.1, [-2.0, -3.0, -0.5])
    assert best["rank"] == 1
    assert abs(best["estimate"] - math.log2(4)) < 1e-12
    with pytest.raises(ValueError):
        kamath_memorization_exposure(-1.0, [])


def test_kmfact_membership_and_predicate():
    out = kamath_factscore(["a", "b"], {"a"})
    assert out["estimate"] == 0.5 and out["unsupported"] == ["b"]
    pred = kamath_factscore(["aa", "b"], lambda c: len(c) > 1)
    assert pred["estimate"] == 0.5 and pred["supported"] == ["aa"]
    with pytest.raises(ValueError):
        kamath_factscore([], {"a"})
