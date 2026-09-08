# morie.fn -- test file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent-route checks for the Kamath Ch 3-6 tranche (km042-km109).

Every expected value is derived here by a route the implementation does
NOT take: hand log-sums, explicit block construction, composition
identities between equations, finite differences, invariances (shift,
scale, swap) and exact counts. A mean-of-inputs stub fails these.
"""

import math

from morie.fn import _array_core as np
import pytest

from morie.fn.km042 import kamath_ch3_prompt_label_mapping
from morie.fn.km043 import kamath_ch3_prompt_softmax_label
from morie.fn.km044 import kamath_ch3_prompt_search_argmax
from morie.fn.km045 import kamath_ch3_dante_cloze
from morie.fn.km046 import kamath_ch3_prefix_prompt_template
from morie.fn.km047 import kamath_ch3_translate_prefix_prompt
from morie.fn.km048 import kamath_ch3_cloze_prompt_template
from morie.fn.km049 import kamath_ch3_top1_prompt_metric
from morie.fn.km050 import kamath_ch3_back_translation_prob
from morie.fn.km051 import kamath_ch3_qa_trigger_template
from morie.fn.km052 import kamath_ch3_t5_template_obj
from morie.fn.km053 import kamath_ch3_prefix_tuning_obj
from morie.fn.km054 import kamath_ch4_series_adapter
from morie.fn.km055 import kamath_ch4_parallel_adapter
from morie.fn.km056 import kamath_ch4_full_finetune_obj
from morie.fn.km057 import kamath_ch4_lora_obj
from morie.fn.km058 import kamath_ch4_lora_forward
from morie.fn.km059 import kamath_ch4_kronecker_product
from morie.fn.km060 import kamath_ch4_krona_efficient
from morie.fn.km061 import kamath_ch4_krona_output
from morie.fn.km062 import kamath_ch4_krona_tuned_weights
from morie.fn.km063 import kamath_ch4_vera_forward
from morie.fn.km064 import kamath_ch4_loftq_objective
from morie.fn.km065 import kamath_ch5_reward_loss_pairwise
from morie.fn.km066 import kamath_ch5_reward_kl_penalty
from morie.fn.km067 import kamath_ch5_rm_bradley_terry
from morie.fn.km068 import kamath_ch5_ppo_loss
from morie.fn.km069 import kamath_ch5_rlhf_objective
from morie.fn.km070 import kamath_ch5_rlhf_optimal_policy
from morie.fn.km071 import kamath_ch5_dpo_reward_optimal
from morie.fn.km072 import kamath_ch5_bradley_terry_pref
from morie.fn.km073 import kamath_ch5_pref_sigmoid_form
from morie.fn.km074 import kamath_ch5_dpo_pref_substituted
from morie.fn.km075 import kamath_ch5_dpo_pref_simplified
from morie.fn.km076 import kamath_ch5_dpo_loss
from morie.fn.km077 import kamath_ch6_factscore
from morie.fn.km078 import kamath_ch6_alignment_function
from morie.fn.km079 import kamath_ch6_alignscore_total_loss
from morie.fn.km080 import kamath_ch6_weat_function
from morie.fn.km081 import kamath_ch6_weat_similarity
from morie.fn.km082 import kamath_ch6_weat_effect_size
from morie.fn.km083 import kamath_ch6_ceat_random_effects
from morie.fn.km084 import kamath_ch6_lpbs_bias
from morie.fn.km085 import kamath_ch6_cbs_variance
from morie.fn.km086 import kamath_ch6_pll
from morie.fn.km087 import kamath_ch6_cps_metric
from morie.fn.km088 import kamath_ch6_cat_metric
from morie.fn.km089 import kamath_ch6_sgs_invariance
from morie.fn.km090 import kamath_ch6_co_occurrence_bias
from morie.fn.km091 import kamath_ch6_demographic_representation
from morie.fn.km092 import kamath_ch6_stereotypical_assoc
from morie.fn.km093 import kamath_ch6_honest_score
from morie.fn.km094 import kamath_ch6_debias_regularizer
from morie.fn.km095 import kamath_ch6_gender_direction
from morie.fn.km096 import kamath_ch6_gender_projection_reg
from morie.fn.km097 import kamath_ch6_ear_entropy_reg
from morie.fn.km098 import kamath_ch6_log_prob_ratio_attr
from morie.fn.km099 import kamath_ch6_emt_metric
from morie.fn.km100 import kamath_ch6_toxicity_probability
from morie.fn.km101 import kamath_ch6_toxic_fraction
from morie.fn.km102 import kamath_ch6_lstm_chain_rule
from morie.fn.km103 import kamath_ch6_lstm_softmax_word
from morie.fn.km104 import kamath_ch6_affect_lm
from morie.fn.km105 import kamath_ch6_gedi_combined_loss
from morie.fn.km106 import kamath_ch6_self_diagnosis_prob
from morie.fn.km107 import kamath_ch6_pii_likelihood
from morie.fn.km108 import kamath_ch6_differential_privacy
from morie.fn.km109 import kamath_ch6_perplexity_leakage

TOL = 1e-12


def lcg(n, seed=7):
    """Deterministic uniforms -- no RNG library, reproducible anywhere."""
    s, out = seed, []
    for _ in range(n):
        s = (1664525 * s + 1013904223) % 2 ** 32
        out.append((s + 0.5) / 2 ** 32)
    return out


# --------------------------------------------------------------- Ch 3

def test_km042_reads_the_mapped_answer_word_not_a_mean():
    M = {"pos": "great", "neg": "terrible"}
    out = kamath_ch3_prompt_label_mapping({"great": 0.7, "terrible": 0.3},
                                          "pos", M)
    assert out["estimate"] == 0.7
    # a mean-of-inputs stub would return 0.5 here
    assert abs(out["estimate"] - 0.5) > 0.1
    assert out["label_probs"] == {"pos": 0.7, "neg": 0.3}


def test_km042_rejects_a_distribution_that_does_not_sum_to_one():
    with pytest.raises(ValueError):
        kamath_ch3_prompt_label_mapping({"great": 0.7, "terrible": 0.5},
                                        "pos", {"pos": "great"})
    with pytest.raises(ValueError):
        kamath_ch3_prompt_label_mapping({"great": 1.0}, "absent",
                                        {"pos": "great"})


def test_km043_matches_a_hand_softmax_over_label_words_only():
    w = {"a": [2.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]}
    h = [1.0, 3.0]
    M = {"la": "a", "lb": "b"}
    out = kamath_ch3_prompt_softmax_label(w, h, M)
    za, zb = 2.0, 3.0           # dot products by hand
    denom = math.exp(za) + math.exp(zb)
    assert abs(out["label_probs"]["la"] - math.exp(za) / denom) < TOL
    # "c" is not a label word, so it must not enter the normaliser
    assert abs(sum(out["label_probs"].values()) - 1.0) < TOL


def test_km043_is_invariant_to_a_shared_logit_shift():
    h = [1.0, 0.0]
    M = {"la": "a", "lb": "b"}
    base = kamath_ch3_prompt_softmax_label(
        {"a": [1.0, 0.0], "b": [0.0, 1.0]}, h, M)["label_probs"]
    # adding the same vector to both w's shifts both logits equally
    shifted = kamath_ch3_prompt_softmax_label(
        {"a": [3.0, 0.0], "b": [2.0, 1.0]}, h, M)["label_probs"]
    assert abs(base["la"] - shifted["la"]) < 1e-12


def test_km044_argmax_survives_a_monotone_rescoring():
    cands = ["aa", "b", "cccc"]
    lin = kamath_ch3_prompt_search_argmax("x [z]", cands, len)
    expo = kamath_ch3_prompt_search_argmax(
        "x [z]", cands, lambda s: math.exp(len(s)))
    assert lin["z_hat"] == expo["z_hat"] == "cccc"
    assert lin["filled_prompt"] == "x cccc"


def test_km044_refuses_an_empty_candidate_set():
    with pytest.raises(ValueError):
        kamath_ch3_prompt_search_argmax("x [z]", [], len)


def test_km045_locates_the_single_mask_and_rejects_others():
    out = kamath_ch3_dante_cloze("Warsaw is the capital of [MASK].")
    assert out["mask_index"] == 5 and out["n"] == 6
    with pytest.raises(ValueError):
        kamath_ch3_dante_cloze("no slot here")
    with pytest.raises(ValueError):
        kamath_ch3_dante_cloze("[MASK] and [MASK]")


def test_km046_to_km048_place_the_slot_where_the_book_does():
    pre = kamath_ch3_prefix_prompt_template("Bad.")["prompt"]
    clz = kamath_ch3_cloze_prompt_template("Bad.")["prompt"]
    tr = kamath_ch3_translate_prefix_prompt("Bad.")["prompt"]
    # prefix: nothing after the slot; cloze: template tokens on both sides
    assert pre.endswith("[z]")
    assert not clz.endswith("[z]") and clz.endswith("movie.")
    assert tr.startswith("Translate") and tr.endswith("[z]")


def test_km046_filling_the_slot_removes_it():
    out = kamath_ch3_prefix_prompt_template("Loved it.", "great")
    assert "[z]" not in out["prompt"] and out["slot_filled"] is True
    assert out["n"] == len(out["prompt"].split())
    with pytest.raises(ValueError):
        kamath_ch3_prefix_prompt_template("", "great")


def test_km049_is_a_proportion_of_exact_top1_hits():
    R = [("x1", "pos"), ("x2", "neg"), ("x3", "pos"), ("x4", "neg")]
    P = lambda x, t: {"pos": 0.6, "neg": 0.4}
    out = kamath_ch3_top1_prompt_metric(R, "T", P)
    assert out["estimate"] == 0.5 and out["n_correct"] == 2
    allright = kamath_ch3_top1_prompt_metric(
        [("x", "pos")], "T", P)
    assert allright["estimate"] == 1.0


def test_km049_rejects_an_empty_set_and_a_bad_distribution():
    with pytest.raises(ValueError):
        kamath_ch3_top1_prompt_metric([], "T", lambda x, t: {"a": 1.0})
    with pytest.raises(ValueError):
        kamath_ch3_top1_prompt_metric(
            [("x", "a")], "T", lambda x, t: {"a": 0.7, "b": 0.7})


def test_km050_is_the_product_of_the_two_legs():
    out = kamath_ch3_back_translation_prob("t", "that", 0.4, 0.5)
    assert abs(out["estimate"] - 0.2) < TOL
    # ranking by round trip is monotone in each leg
    worse = kamath_ch3_back_translation_prob("t", "that", 0.4, 0.25)
    assert worse["estimate"] < out["estimate"]
    with pytest.raises(ValueError):
        kamath_ch3_back_translation_prob("t", "that", 0.4)


def test_km051_repeats_the_trigger_exactly_n_times():
    out = kamath_ch3_qa_trigger_template("Q?", "C.", "zz", "adv",
                                         n_triggers=5)
    assert out["prompt"].split().count("zz") == 5
    assert out["prompt"].endswith("adv")
    with pytest.raises(ValueError):
        kamath_ch3_qa_trigger_template("Q?", "C.", "zz", "adv",
                                       n_triggers=0)


def test_km052_equals_a_hand_log_sum():
    D = [("a", "pos"), ("b", "neg"), ("c", "pos")]
    probs = {"a": 0.5, "b": 0.25, "c": 0.125}
    T5 = lambda T, s: probs[s.split()[0]]
    out = kamath_ch3_t5_template_obj(D, "{x} -> {y}", T5)
    hand = math.log(0.5) + math.log(0.25) + math.log(0.125)
    assert abs(out["estimate"] - hand) < TOL
    # a mean would be a third of this
    assert abs(out["estimate"] - hand / 3) > 1.0


def test_km052_rejects_impossible_templates():
    with pytest.raises(ValueError):
        kamath_ch3_t5_template_obj([("a", "b")], "{x}{y}",
                                   lambda T, s: 0.0)


def test_km053_sums_only_over_the_index_set():
    p = {"a": 0.5, "b": 0.25, "c": 0.125}
    phi = lambda z, hp: p[z]
    y, h = ["a", "b", "c"], [0.0, 1.0, 2.0]
    full = kamath_ch3_prefix_tuning_obj(phi, "x", y, h)
    part = kamath_ch3_prefix_tuning_obj(phi, "x", y, h, Y_idx=[0, 2])
    assert abs(full["estimate"] -
               (math.log(0.5) + math.log(0.25) + math.log(0.125))) < TOL
    assert abs(part["estimate"] -
               (math.log(0.5) + math.log(0.125))) < TOL


def test_km053_sees_the_prefix_h_before_the_position():
    seen = []
    phi = lambda z, hp: (seen.append(len(hp)), 0.5)[1]
    kamath_ch3_prefix_tuning_obj(phi, "x", ["a", "b", "c"],
                                 [0.0, 1.0, 2.0])
    assert seen == [0, 1, 2]
    with pytest.raises(ValueError):
        kamath_ch3_prefix_tuning_obj(phi, "x", ["a"], [0.0, 1.0])


# --------------------------------------------------------------- Ch 4

def test_km054_matches_a_hand_computed_bottleneck():
    H = [[1.0, -2.0]]
    Wd = [[1.0, 0.0], [0.0, 1.0]]
    Wu = [[1.0, 1.0], [1.0, 1.0]]
    out = kamath_ch4_series_adapter(H, Wd, Wu)
    # relu([1, -2]) = [1, 0]; [1,0] @ Wu = [1, 1]
    assert out["output"] == [[2.0, -1.0]]
    assert out["delta"] == [[1.0, 1.0]]


def test_km054_zero_up_projection_is_the_identity():
    H = [[3.0, 4.0]]
    out = kamath_ch4_series_adapter(H, [[1.0], [1.0]], [[0.0, 0.0]])
    assert out["output"] == H


def test_km055_reduces_to_km054_when_the_input_is_the_output():
    H = [[1.0, -2.0]]
    Wd = [[1.0, 0.0], [0.0, 1.0]]
    Wu = [[1.0, 1.0], [1.0, 1.0]]
    ser = kamath_ch4_series_adapter(H, Wd, Wu)
    par = kamath_ch4_parallel_adapter(H, H, Wd, Wu)
    assert par["output"] == ser["output"]
    other = kamath_ch4_parallel_adapter(H, [[0.0, 0.0]], Wd, Wu)
    assert other["output"] == H       # a different branch input matters


def test_km055_shape_errors_are_specific():
    with pytest.raises(ValueError):
        kamath_ch4_parallel_adapter([[1.0, 2.0]], [[1.0]],
                                    [[1.0], [1.0]], [[1.0, 1.0]])


def test_km056_equals_a_hand_double_sum():
    table = {("d1", "a"): 0.5, ("d1", "b"): 0.25, ("d2", "c"): 0.125}
    model = lambda xi, pre, t: table[(xi, t)]
    out = kamath_ch4_full_finetune_obj(model, ["d1", "d2"],
                                       [["a", "b"], ["c"]])
    hand = math.log(0.5) + math.log(0.25) + math.log(0.125)
    assert abs(out["estimate"] - hand) < TOL
    assert out["n_tokens"] == 3 and out["n"] == 2
    assert abs(out["per_pair"][1] - math.log(0.125)) < TOL


def test_km056_rejects_mismatched_and_empty_inputs():
    m = lambda xi, pre, t: 0.5
    with pytest.raises(ValueError):
        kamath_ch4_full_finetune_obj(m, ["a"], [["u"], ["v"]])
    with pytest.raises(ValueError):
        kamath_ch4_full_finetune_obj(m, [], [])


def test_km057_scores_the_adapted_model_against_the_frozen_one():
    adapted = lambda xi, pre, t: 0.5
    base = lambda xi, pre, t: 0.125
    x, y = ["d"], [["a", "b"]]
    out = kamath_ch4_lora_obj(adapted, base, x, y)
    ref = kamath_ch4_full_finetune_obj(adapted, x, y)
    assert abs(out["estimate"] - ref["estimate"]) < TOL
    assert abs(out["improvement"] - 2 * math.log(4.0)) < TOL


def test_km057_improvement_is_zero_when_the_adapter_changes_nothing():
    same = lambda xi, pre, t: 0.3
    out = kamath_ch4_lora_obj(same, same, ["d"], [["a"]])
    assert abs(out["improvement"]) < TOL


def test_km058_is_linear_in_the_input():
    W0 = [[1.0, 2.0], [0.0, 1.0]]
    B = [[1.0], [2.0]]
    A = [[1.0, -1.0]]
    x1, x2 = [1.0, 0.0], [0.0, 3.0]
    h1 = np.array(kamath_ch4_lora_forward(W0, B, A, x1)["h"])
    h2 = np.array(kamath_ch4_lora_forward(W0, B, A, x2)["h"])
    hs = np.array(kamath_ch4_lora_forward(
        W0, B, A, [1.0, 3.0])["h"])
    assert np.allclose(h1 + h2, hs)


def test_km058_delta_has_rank_at_most_r():
    out = kamath_ch4_lora_forward([[1.0, 0.0], [0.0, 1.0]],
                                  [[1.0], [2.0]], [[1.0, 1.0]],
                                  [1.0, 1.0])
    assert out["r"] == 1 and out["delta_W_rank"] <= 1
    assert out["delta_h"] == [2.0, 4.0]
    with pytest.raises(ValueError):
        kamath_ch4_lora_forward([[1.0, 0.0]], [[1.0]], [[1.0]],
                                [1.0, 0.0])


def test_km059_matches_an_explicit_block_construction():
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[0.0, 5.0], [6.0, 7.0]]
    got = np.array(kamath_ch4_kronecker_product(A, B)["W"])
    Am, Bm = np.array(A), np.array(B)
    hand = np.zeros((4, 4))
    for i in range(2):
        for j in range(2):
            hand[2 * i:2 * i + 2, 2 * j:2 * j + 2] = Am[i, j] * Bm
    assert np.allclose(got, hand)


def test_km059_rank_multiplies():
    out = kamath_ch4_kronecker_product([[1.0, 2.0], [2.0, 4.0]],
                                       [[1.0, 0.0], [0.0, 1.0]])
    assert out["rank"] == 1 * 2      # rank(A)=1, rank(B)=2
    assert out["shape"] == (4, 4) and out["n_params"] == 8


def test_km060_agrees_with_forming_the_full_kronecker_matrix():
    u = lcg(8)
    A = np.array(u[:4]).reshape(2, 2)
    B = np.array(u[4:8]).reshape(2, 2)
    x = np.array(lcg(4, seed=99))
    got = np.array(kamath_ch4_krona_efficient(A, B, x)["y"])
    hand = np.kron(A, B) @ x
    assert np.allclose(got, hand)


def test_km060_handles_non_square_factors_and_bad_lengths():
    A = np.array([[1.0, 2.0, 3.0]])          # 1 x 3
    B = np.array([[1.0, 0.0], [0.0, 2.0]])   # 2 x 2
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    got = np.array(kamath_ch4_krona_efficient(A, B, x)["y"])
    assert np.allclose(got, np.kron(A, B) @ x)
    with pytest.raises(ValueError):
        kamath_ch4_krona_efficient(A, B, [1.0, 2.0])


def test_km061_equals_x_times_the_merged_weight_of_km062():
    X = [[1.0, -1.0], [2.0, 0.5]]
    W = [[1.0, 0.0], [0.0, 1.0]]
    Ak = [[1.0, 2.0], [0.0, 1.0]]
    Bk = [[3.0]]
    y = np.array(kamath_ch4_krona_output(X, W, Ak, Bk, 0.5)["Y"])
    Wt = np.array(kamath_ch4_krona_tuned_weights(W, Ak, Bk, 0.5)["W_tuned"])
    assert np.allclose(y, np.array(X) @ Wt)


def test_km061_and_km062_are_the_identity_at_s_zero():
    X = [[1.0, 2.0]]
    W = [[1.0, 0.0], [0.0, 1.0]]
    Ak, Bk = [[5.0, 5.0], [5.0, 5.0]], [[7.0]]
    assert kamath_ch4_krona_tuned_weights(W, Ak, Bk, 0.0)["W_tuned"] == W
    assert np.allclose(kamath_ch4_krona_output(X, W, Ak, Bk, 0.0)["Y"],
                       np.array(X) @ np.array(W))


def test_km062_rejects_an_adapter_of_the_wrong_shape():
    with pytest.raises(ValueError):
        kamath_ch4_krona_tuned_weights([[1.0, 0.0], [0.0, 1.0]],
                                       [[1.0]], [[1.0]], 1.0)


def test_km063_update_is_linear_in_each_diagonal():
    W0 = [[1.0, 0.0], [0.0, 1.0]]
    A = [[1.0, 1.0]]
    B = [[2.0], [1.0]]
    x = [1.0, 1.0]
    d1 = np.array(kamath_ch4_vera_forward(W0, [1.0, 1.0], [1.0], A, B,
                                          x)["delta_h"])
    d2 = np.array(kamath_ch4_vera_forward(W0, [1.0, 1.0], [3.0], A, B,
                                          x)["delta_h"])
    assert np.allclose(d2, 3.0 * d1)
    d3 = np.array(kamath_ch4_vera_forward(W0, [2.0, 5.0], [1.0], A, B,
                                          x)["delta_h"])
    assert np.allclose(d3, np.array([2.0, 5.0]) * d1)


def test_km063_parameter_counts_beat_lora():
    out = kamath_ch4_vera_forward([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0],
                                  [1.0], [[1.0, 0.0]], [[1.0], [0.0]],
                                  [1.0, 1.0])
    assert out["n_trainable"] == 3 and out["n_trainable_lora"] == 4
    with pytest.raises(ValueError):
        kamath_ch4_vera_forward([[1.0, 0.0], [0.0, 1.0]], [1.0],
                                [1.0], [[1.0, 0.0]], [[1.0], [0.0]],
                                [1.0, 1.0])


def test_km064_equals_a_hand_frobenius_norm():
    W = [[1.0, 2.0], [3.0, 4.0]]
    Q = [[1.0, 0.0], [0.0, 0.0]]
    A = [[1.0], [0.0]]
    B = [[0.0], [1.0]]
    out = kamath_ch4_loftq_objective(W, Q, A, B)
    resid = np.array(W) - np.array(Q) - np.array(A) @ np.array(B).T
    hand = math.sqrt(float(np.sum(resid ** 2)))
    assert abs(out["estimate"] - hand) < 1e-12
    assert out["estimate"] > 0


def test_km064_is_zero_for_an_exact_decomposition():
    W = [[2.0, 0.0], [0.0, 3.0]]
    Q = [[2.0, 0.0], [0.0, 0.0]]
    A = [[0.0], [3.0]]
    B = [[0.0], [1.0]]
    assert kamath_ch4_loftq_objective(W, Q, A, B)["estimate"] == 0.0


# --------------------------------------------------------------- Ch 5

def test_km065_matches_a_hand_softplus_and_ignores_reward_shifts():
    r = lambda x, y: {"g": 2.0, "b": 0.5}[y]
    out = kamath_ch5_reward_loss_pairwise(r, ["p"], ["g"], ["b"], [0])
    assert abs(out["estimate"] - math.log(1 + math.exp(-1.5))) < TOL
    shifted = kamath_ch5_reward_loss_pairwise(
        lambda x, y: r(x, y) + 100.0, ["p"], ["g"], ["b"], [0])
    assert abs(shifted["estimate"] - out["estimate"]) < 1e-10


def test_km065_gradient_matches_a_finite_difference():
    # d/dm of -log sigmoid(m) is -sigmoid(-m)
    def loss(m):
        r = lambda x, y: m if y == "w" else 0.0
        return kamath_ch5_reward_loss_pairwise(
            r, ["p"], ["w"], ["l"], [0])["estimate"]
    m, eps = 0.75, 1e-6
    fd = (loss(m + eps) - loss(m - eps)) / (2 * eps)
    analytic = -1.0 / (1.0 + math.exp(m))
    assert abs(fd - analytic) < 1e-7


def test_km065_honours_the_preference_index():
    r = lambda x, y: {"a": 3.0, "b": 0.0}[y]
    first = kamath_ch5_reward_loss_pairwise(r, ["p"], ["a"], ["b"], [0])
    second = kamath_ch5_reward_loss_pairwise(r, ["p"], ["a"], ["b"], [1])
    assert first["margins"] == [3.0] and second["margins"] == [-3.0]
    assert second["estimate"] > first["estimate"]


def test_km066_matches_the_hand_penalty_and_zero_beta():
    out = kamath_ch5_reward_kl_penalty("p", "y", 0.8, 0.2, 0.5,
                                       r_theta=2.0)
    assert abs(out["estimate"] - (2.0 - 0.5 * math.log(4.0))) < TOL
    plain = kamath_ch5_reward_kl_penalty("p", "y", 0.8, 0.2, 0.0,
                                         r_theta=2.0)
    assert plain["estimate"] == 2.0


def test_km066_requires_a_reward_and_valid_probabilities():
    with pytest.raises(ValueError):
        kamath_ch5_reward_kl_penalty("p", "y", 0.5, 0.5, 1.0)
    with pytest.raises(ValueError):
        kamath_ch5_reward_kl_penalty("p", "y", 0.0, 0.5, 1.0, r_theta=1.0)


def test_km067_agrees_with_km065_pair_for_pair():
    r = lambda x, y: {"w": 1.25, "l": -0.5}[y]
    bt = kamath_ch5_rm_bradley_terry(["p", "q"], ["w", "w"], ["l", "l"], r)
    pair = kamath_ch5_reward_loss_pairwise(r, ["p", "q"], ["w", "w"],
                                           ["l", "l"], [0, 0])
    assert abs(bt["estimate"] - pair["estimate"]) < TOL
    assert abs(bt["estimate"] - math.log(1 + math.exp(-1.75))) < TOL


def test_km068_is_the_negated_mean_of_km069():
    r = lambda x, y: {"a": 1.0, "b": 0.0}[y]
    phi = [[0.75, 0.25], [0.5, 0.5]]
    ref = [[0.5, 0.5], [0.5, 0.5]]
    ppo = kamath_ch5_ppo_loss(phi, ["p", "q"], [["a", "b"], ["a", "b"]],
                              r, 0.5, pi_ref=ref)
    j = [kamath_ch5_rlhf_objective(p, q, [1.0, 0.0], 0.5)["estimate"]
         for p, q in zip(phi, ref)]
    assert abs(ppo["estimate"] + sum(j) / 2) < TOL


def test_km068_requires_the_reference_policy():
    with pytest.raises(ValueError):
        kamath_ch5_ppo_loss([[0.5, 0.5]], ["p"], [["a", "b"]],
                            lambda x, y: 1.0, 1.0)


def test_km069_kl_matches_a_hand_computation_and_penalises():
    p, q = [0.75, 0.25], [0.5, 0.5]
    hand_kl = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
    out = kamath_ch5_rlhf_objective(p, q, [1.0, 0.0], 2.0)
    assert abs(out["kl"] - hand_kl) < TOL
    assert abs(out["expected_reward"] - 0.75) < TOL
    assert abs(out["estimate"] - (0.75 - 2.0 * hand_kl)) < TOL
    free = kamath_ch5_rlhf_objective(p, q, [1.0, 0.0], 0.0)
    assert free["estimate"] > out["estimate"]


def test_km069_rejects_non_distributions_and_infinite_kl():
    with pytest.raises(ValueError):
        kamath_ch5_rlhf_objective([0.6, 0.6], [0.5, 0.5], [1.0, 0.0], 1.0)
    with pytest.raises(ValueError):
        kamath_ch5_rlhf_objective([0.5, 0.5], [1.0, 0.0], [1.0, 0.0], 1.0)


def test_km070_returns_a_distribution_and_relaxes_to_pi_ref():
    out = kamath_ch5_rlhf_optimal_policy([0.25, 0.75], [1.0, 0.0], 1.0)
    assert abs(sum(out["pi"]) - 1.0) < TOL
    hand0 = 0.25 * math.e / (0.25 * math.e + 0.75)
    assert abs(out["pi"][0] - hand0) < TOL
    cold = kamath_ch5_rlhf_optimal_policy([0.25, 0.75], [1.0, 0.0], 0.01)
    warm = kamath_ch5_rlhf_optimal_policy([0.25, 0.75], [1.0, 0.0], 1e6)
    assert cold["pi"][0] > 0.99 and abs(warm["pi"][0] - 0.25) < 1e-4


def test_km070_checks_a_supplied_partition_function():
    with pytest.raises(ValueError):
        kamath_ch5_rlhf_optimal_policy([0.5, 0.5], [1.0, 0.0], 1.0, Z=1.0)
    with pytest.raises(ValueError):
        kamath_ch5_rlhf_optimal_policy([0.5, 0.5], [1.0, 0.0], 0.0)


def test_km071_inverts_km070():
    q, r, beta = [0.25, 0.75], [1.5, -0.5], 0.8
    pol = kamath_ch5_rlhf_optimal_policy(q, r, beta)
    back = kamath_ch5_dpo_reward_optimal(pol["pi"], q, beta, Z=pol["Z"])
    assert np.allclose(back["r"], r)


def test_km071_offset_shifts_every_reward_equally():
    a = kamath_ch5_dpo_reward_optimal([0.5, 0.5], [0.25, 0.75], 2.0)
    b = kamath_ch5_dpo_reward_optimal([0.5, 0.5], [0.25, 0.75], 2.0,
                                      Z=math.e)
    assert np.allclose(np.array(b["r"]) - np.array(a["r"]), 2.0)


def test_km072_is_symmetric_and_hand_checkable():
    r = {"a": 1.5, "b": -0.5}
    ab = kamath_ch5_bradley_terry_pref(r, "a", "b")["estimate"]
    ba = kamath_ch5_bradley_terry_pref(r, "b", "a")["estimate"]
    assert abs(ab + ba - 1.0) < TOL
    hand = math.exp(1.5) / (math.exp(1.5) + math.exp(-0.5))
    assert abs(ab - hand) < TOL


def test_km072_survives_rewards_that_would_overflow_the_ratio_form():
    p = kamath_ch5_bradley_terry_pref({"a": 800.0, "b": 0.0}, "a", "b")
    assert p["estimate"] == 1.0 and math.isfinite(p["margin"])


def test_km073_reproduces_km072():
    r = {"a": 0.3, "b": -1.2}
    assert abs(kamath_ch5_pref_sigmoid_form([0.3, -1.2])["estimate"] -
               kamath_ch5_bradley_terry_pref(r, "a", "b")["estimate"]) < TOL
    with pytest.raises(ValueError):
        kamath_ch5_pref_sigmoid_form([1.0, 2.0, 3.0])


def test_km074_is_independent_of_z():
    args = ([0.6, 0.2], [0.3, 0.4], 1.5)
    a = kamath_ch5_dpo_pref_substituted(*args, Z=1.0)["estimate"]
    b = kamath_ch5_dpo_pref_substituted(*args, Z=1e6)["estimate"]
    c = kamath_ch5_dpo_pref_simplified(*args)["estimate"]
    assert abs(a - b) < TOL and abs(a - c) < TOL


def test_km075_matches_a_hand_sigmoid_of_log_ratios():
    beta = 2.0
    ps, pr = [0.6, 0.2], [0.3, 0.4]
    m = beta * (math.log(0.6 / 0.3) - math.log(0.2 / 0.4))
    out = kamath_ch5_dpo_pref_simplified(ps, pr, beta)
    assert abs(out["margin"] - m) < TOL
    assert abs(out["estimate"] - 1 / (1 + math.exp(-m))) < TOL


def test_km075_is_one_half_when_the_policy_is_the_reference():
    assert kamath_ch5_dpo_pref_simplified([0.3, 0.7], [0.3, 0.7],
                                          4.0)["estimate"] == 0.5
    with pytest.raises(ValueError):
        kamath_ch5_dpo_pref_simplified([0.3, 0.7], [0.3, 0.7], -1.0)


def test_km076_is_minus_mean_log_of_km075():
    pt = [[0.6, 0.2], [0.4, 0.4]]
    pr = [[0.3, 0.4], [0.5, 0.2]]
    beta = 1.25
    loss = kamath_ch5_dpo_loss(pt, pr, beta)["estimate"]
    hand = -sum(math.log(kamath_ch5_dpo_pref_simplified(a, b, beta)
                         ["estimate"]) for a, b in zip(pt, pr)) / 2
    assert abs(loss - hand) < 1e-12


def test_km076_falls_as_the_winner_gains_probability():
    ref = [[0.5, 0.5]]
    worse = kamath_ch5_dpo_loss([[0.5, 0.5]], ref, 1.0)["estimate"]
    better = kamath_ch5_dpo_loss([[0.9, 0.1]], ref, 1.0)["estimate"]
    assert better < worse
    assert abs(worse - math.log(2.0)) < TOL


# --------------------------------------------------------------- Ch 6

def test_km077_conditions_on_the_model_responding():
    M = lambda x: None if x == "skip" else x
    out = kamath_ch6_factscore(M, ["a b c d", "skip", "a"], str.split,
                               {"a"})
    # scored prompts: 1/4 and 1/1 -> mean 0.625, abstention excluded
    assert abs(out["estimate"] - 0.625) < TOL
    assert out["n_responded"] == 2 and out["n"] == 3


def test_km077_refuses_a_response_with_no_atomic_facts():
    with pytest.raises(ValueError):
        kamath_ch6_factscore(lambda x: x, ["a"], lambda y: [], {"a"})
    with pytest.raises(ValueError):
        kamath_ch6_factscore(lambda x: None, ["a"], str.split, {"a"})


def test_km078_enforces_the_declared_label_space():
    ok = kamath_ch6_alignment_function("a", "b", "bin",
                                       f=lambda a, b: "NOT ALIGNED")
    assert ok["estimate"] == 0.0 and ok["label"] == "NOT ALIGNED"
    with pytest.raises(ValueError):
        kamath_ch6_alignment_function("a", "b", "bin",
                                      f=lambda a, b: "CONTRADICT")
    with pytest.raises(ValueError):
        kamath_ch6_alignment_function("a", "b", "reg",
                                      f=lambda a, b: 1.5)


def test_km078_has_no_default_alignment_function():
    with pytest.raises(ValueError):
        kamath_ch6_alignment_function("a", "b", "bin")


def test_km079_is_linear_in_each_component():
    base = kamath_ch6_alignscore_total_loss(1.0, 1.0, 1.0,
                                            [0.2, 0.3, 0.5])["estimate"]
    bumped = kamath_ch6_alignscore_total_loss(2.0, 1.0, 1.0,
                                              [0.2, 0.3, 0.5])["estimate"]
    assert abs(base - 1.0) < TOL and abs(bumped - base - 0.2) < TOL


def test_km079_rejects_bad_weight_vectors():
    with pytest.raises(ValueError):
        kamath_ch6_alignscore_total_loss(1.0, 1.0, 1.0, [0.5, 0.5])
    with pytest.raises(ValueError):
        kamath_ch6_alignscore_total_loss(1.0, 1.0, 1.0, [-1.0, 1.0, 1.0])


def test_km081_matches_hand_cosines():
    a = [3.0, 4.0]                       # norm 5
    W1 = [[5.0, 0.0], [0.0, 5.0]]        # cosines 0.6 and 0.8
    W2 = [[-1.0, 0.0]]                   # cosine -0.6
    out = kamath_ch6_weat_similarity(a, W1, W2)
    assert abs(out["mean_cos_W1"] - 0.7) < 1e-12
    assert abs(out["estimate"] - (0.7 + 0.6)) < 1e-12


def test_km081_is_scale_invariant_and_rejects_zero_vectors():
    base = kamath_ch6_weat_similarity([1.0, 2.0], [[1.0, 0.0]],
                                      [[0.0, 1.0]])["estimate"]
    scaled = kamath_ch6_weat_similarity([10.0, 20.0], [[7.0, 0.0]],
                                        [[0.0, 3.0]])["estimate"]
    assert abs(base - scaled) < 1e-12
    with pytest.raises(ValueError):
        kamath_ch6_weat_similarity([1.0, 0.0], [[0.0, 0.0]], [[0.0, 1.0]])


def test_km080_negates_when_the_protected_sets_swap():
    A1, A2 = [[1.0, 0.0], [2.0, 1.0]], [[0.0, 1.0]]
    W1, W2 = [[1.0, 0.0]], [[0.0, 1.0]]
    f = kamath_ch6_weat_function(A1, A2, W1, W2)["estimate"]
    g = kamath_ch6_weat_function(A2, A1, W1, W2)["estimate"]
    assert abs(f + g) < 1e-12
    assert f > 0


def test_km080_sums_rather_than_averages():
    A1 = [[1.0, 0.0], [1.0, 0.0]]
    A2 = [[0.0, 1.0]]
    out = kamath_ch6_weat_function(A1, A2, [[1.0, 0.0]], [[0.0, 1.0]])
    assert abs(out["estimate"] - (1.0 + 1.0 - (-1.0))) < 1e-12


def test_km082_is_invariant_to_scaling_every_embedding():
    A1, A2 = [[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [-1.0, 1.0]]
    W1, W2 = [[1.0, 0.0]], [[0.0, 1.0]]
    a = kamath_ch6_weat_effect_size(A1, A2, W1, W2)["estimate"]
    b = kamath_ch6_weat_effect_size(
        (np.array(A1) * 5).tolist(), (np.array(A2) * 5).tolist(),
        W1, W2)["estimate"]
    assert abs(a - b) < 1e-12


def test_km082_matches_a_hand_standardisation():
    A1, A2 = [[1.0, 0.0]], [[0.0, 1.0]]
    W1, W2 = [[1.0, 0.0]], [[0.0, 1.0]]
    out = kamath_ch6_weat_effect_size(A1, A2, W1, W2)
    s = np.array([1.0, -1.0])
    assert abs(out["estimate"] - (1.0 - (-1.0)) / s.std()) < 1e-12
    with pytest.raises(ValueError):
        kamath_ch6_weat_effect_size(A1, A1, W1, W2)   # zero spread


def test_km083_equal_weights_give_the_plain_mean():
    A1 = [[[1.0, 0.0]], [[1.0, 1.0]]]
    A2 = [[[0.0, 1.0]], [[0.0, 1.0]]]
    W1 = [[[1.0, 0.0]], [[1.0, 0.0]]]
    W2 = [[[0.0, 1.0]], [[0.0, 1.0]]]
    pooled = kamath_ch6_ceat_random_effects(A1, A2, W1, W2, [1.0, 1.0])
    assert abs(pooled["estimate"] - float(np.mean(pooled["weat"]))) < 1e-12


def test_km083_weights_pull_toward_the_heavier_sample():
    A1 = [[[1.0, 0.0]], [[0.0, 1.0]]]
    A2 = [[[0.0, 1.0]], [[1.0, 0.0]]]
    W1 = [[[1.0, 0.0]], [[1.0, 0.0]]]
    W2 = [[[0.0, 1.0]], [[0.0, 1.0]]]
    out = kamath_ch6_ceat_random_effects(A1, A2, W1, W2, [3.0, 1.0])
    w = np.array([3.0, 1.0])
    assert abs(out["estimate"] -
               float(np.sum(w * np.array(out["weat"])) / w.sum())) < 1e-12
    with pytest.raises(ValueError):
        kamath_ch6_ceat_random_effects(A1, A2, W1, W2, [0.0, 0.0])


def test_km084_is_antisymmetric_and_hand_checkable():
    ab = kamath_ch6_lpbs_bias([0.4, 0.1], [0.2, 0.2])["estimate"]
    ba = kamath_ch6_lpbs_bias([0.1, 0.4], [0.2, 0.2])["estimate"]
    assert abs(ab - (math.log(2.0) - math.log(0.5))) < TOL
    assert abs(ab + ba) < TOL


def test_km084_the_prior_normalisation_matters():
    # equal raw probabilities, unequal priors -> non-zero bias
    out = kamath_ch6_lpbs_bias([0.3, 0.3], [0.1, 0.6])
    assert out["estimate"] > 0
    assert kamath_ch6_lpbs_bias([0.3, 0.3], [0.2, 0.2])["estimate"] == 0.0


def test_km085_equals_a_hand_variance_and_is_never_negative():
    p_a = [[0.4, 0.1, 0.2], [0.3, 0.3, 0.3]]
    p_pr = [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3]]
    out = kamath_ch6_cbs_variance(["w1", "w2"], ["a", "b", "c"], p_a, p_pr)
    logs = np.log(np.array(p_a) / np.array(p_pr))
    hand = float(np.mean(np.var(logs, axis=1)))
    assert abs(out["estimate"] - hand) < 1e-12
    assert out["estimate"] >= 0 and out["per_word"][1] == 0.0


def test_km085_needs_at_least_two_groups():
    with pytest.raises(ValueError):
        kamath_ch6_cbs_variance(["w"], ["a"], [[0.5]], [[0.5]])


def test_km086_equals_a_hand_log_sum():
    p = [0.5, 0.2, 0.8]
    out = kamath_ch6_pll(p)
    assert abs(out["estimate"] - sum(math.log(v) for v in p)) < 1e-12
    # never a mean of the inputs
    assert abs(out["estimate"] - float(np.mean(p))) > 1.0


def test_km086_uses_the_scorer_when_given_and_rejects_zeros():
    probs = [0.5, 0.25]
    out = kamath_ch6_pll(["x", "y"], theta=lambda i: probs[i])
    assert abs(out["estimate"] + math.log(8.0)) < TOL
    with pytest.raises(ValueError):
        kamath_ch6_pll([0.5, 0.0])


def test_km087_matches_km086_over_the_same_probabilities():
    probs = [0.5, 0.25, 0.5]
    cps = kamath_ch6_cps_metric(["u1", "u2", "u3"], ["he"],
                                theta=lambda U, M, i: probs[i])
    assert abs(cps["estimate"] - kamath_ch6_pll(probs)["estimate"]) < TOL
    with pytest.raises(ValueError):
        kamath_ch6_cps_metric([0.5], [])


def test_km088_averages_where_km087_sums():
    probs = [0.5, 0.25, 0.5]
    cat = kamath_ch6_cat_metric(probs, ["ctx"])
    pll = kamath_ch6_pll(probs)
    assert abs(cat["estimate"] - pll["estimate"] / 3) < 1e-12
    assert cat["n_context"] == 1


def test_km088_requires_context_tokens():
    with pytest.raises(ValueError):
        kamath_ch6_cat_metric([0.5], [])


def test_km089_counts_exact_matches_by_default():
    out = kamath_ch6_sgs_invariance(["a", "b", "c", "d"],
                                    ["a", "x", "c", "y"])
    assert out["estimate"] == 0.5 and out["n_invariant"] == 2


def test_km089_rejects_unpaired_outputs_and_bad_psi():
    with pytest.raises(ValueError):
        kamath_ch6_sgs_invariance(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        kamath_ch6_sgs_invariance(["a"], ["b"], psi=lambda u, v: 7.0)


def test_km090_flips_sign_when_the_attribute_sets_swap():
    Ai = ["nurse nurse doctor nurse"]
    Aj = ["nurse doctor doctor doctor"]
    f = kamath_ch6_co_occurrence_bias("nurse", Ai, Aj)["estimate"]
    g = kamath_ch6_co_occurrence_bias("nurse", Aj, Ai)["estimate"]
    assert abs(f - math.log(3.0)) < 1e-12 and abs(f + g) < 1e-12


def test_km090_refuses_an_absent_token():
    with pytest.raises(ValueError):
        kamath_ch6_co_occurrence_bias("nurse", ["doctor"], ["nurse"])


def test_km091_counts_every_occurrence_not_every_output():
    out = kamath_ch6_demographic_representation(
        "fem", ["she", "her"], ["she and she", "her", "he"])
    assert out["estimate"] == 3.0
    assert out["per_word"] == {"she": 2, "her": 1}
    assert out["n"] == 5


def test_km091_rejects_empty_inputs():
    with pytest.raises(ValueError):
        kamath_ch6_demographic_representation("g", [], ["a"])
    with pytest.raises(ValueError):
        kamath_ch6_demographic_representation("g", ["a"], [])


def test_km092_gates_whole_outputs_on_the_stereotyped_word():
    Y = ["she she nurse nurse", "she doctor", "nurse"]
    out = kamath_ch6_stereotypical_assoc("nurse", ["she"], Y)
    # only the first output has both; it contributes C(she) = 2
    assert out["estimate"] == 2.0 and out["n_outputs_with_w"] == 2


def test_km092_is_bounded_by_km091():
    Y = ["she nurse", "she doctor"]
    st = kamath_ch6_stereotypical_assoc("nurse", ["she"], Y)["estimate"]
    dr = kamath_ch6_demographic_representation("g", ["she"], Y)["estimate"]
    assert st <= dr and st == 1.0


def test_km093_denominator_is_prompts_times_k():
    Y = [["a", "bad"], ["bad", "bad"], ["ok", "ok"]]
    out = kamath_ch6_honest_score(Y, 2, hurtlex={"bad"})
    assert out["n_hurtful"] == 3 and out["n_completions"] == 6
    assert abs(out["estimate"] - 0.5) < TOL


def test_km093_requires_exactly_k_completions_and_a_lexicon():
    with pytest.raises(ValueError):
        kamath_ch6_honest_score([["a", "b"], ["c"]], 2, hurtlex={"a"})
    with pytest.raises(ValueError):
        kamath_ch6_honest_score([["a"]], 1)


def test_km094_is_zero_only_for_identical_embeddings():
    E = {"a": [1.0, 2.0], "b": [1.0, 2.0], "c": [4.0, 6.0]}
    assert kamath_ch6_debias_regularizer([("a", "b")], E, 3.0)["estimate"] \
        == 0.0
    out = kamath_ch6_debias_regularizer([("a", "c")], E, 3.0)
    assert abs(out["estimate"] - 3.0 * (9.0 + 16.0)) < 1e-12


def test_km094_scales_linearly_with_lam():
    E = {"a": [0.0, 0.0], "b": [1.0, 1.0]}
    one = kamath_ch6_debias_regularizer([("a", "b")], E, 1.0)["estimate"]
    ten = kamath_ch6_debias_regularizer([("a", "b")], E, 10.0)["estimate"]
    assert abs(ten - 10.0 * one) < 1e-12
    with pytest.raises(ValueError):
        kamath_ch6_debias_regularizer([], E, 1.0)


def test_km095_averages_the_pairwise_displacements():
    E = {"f1": [0.0, 0.0], "m1": [2.0, 0.0],
         "f2": [1.0, 1.0], "m2": [1.0, 5.0]}
    out = kamath_ch6_gender_direction([("f1", "m1"), ("f2", "m2")], E)
    assert out["g"] == [1.0, 2.0]


def test_km095_negates_when_the_pair_order_flips():
    E = {"f": [0.0, 1.0], "m": [3.0, 1.0]}
    a = kamath_ch6_gender_direction([("f", "m")], E)["g"]
    b = kamath_ch6_gender_direction([("m", "f")], E)["g"]
    assert np.allclose(np.array(a), -np.array(b))


def test_km096_is_zero_for_orthogonal_embeddings():
    out = kamath_ch6_gender_projection_reg([[0.0, 3.0], [0.0, -4.0]],
                                           [5.0, 0.0])
    assert out["estimate"] == 0.0 and out["sum_abs"] == 0.0


def test_km096_is_invariant_to_the_length_of_g():
    W = [[1.0, 1.0], [2.0, -3.0]]
    a = kamath_ch6_gender_projection_reg(W, [1.0, 0.0])["estimate"]
    b = kamath_ch6_gender_projection_reg(W, [100.0, 0.0])["estimate"]
    assert abs(a - 3.0) < 1e-12 and abs(a - b) < 1e-12
    with pytest.raises(ValueError):
        kamath_ch6_gender_projection_reg(W, [0.0, 0.0])


def test_km097_uniform_attention_maximises_entropy():
    uni = [[[0.25] * 4]]
    onehot = [[[1.0, 0.0, 0.0, 0.0]]]
    u = kamath_ch6_ear_entropy_reg(uni, lam=1.0)
    o = kamath_ch6_ear_entropy_reg(onehot, lam=1.0)
    assert abs(u["total_entropy"] - math.log(4.0)) < 1e-12
    assert o["total_entropy"] == 0.0 and u["estimate"] < o["estimate"]


def test_km097_sums_over_layers_and_checks_the_rows():
    two = kamath_ch6_ear_entropy_reg([[[0.5, 0.5]], [[0.5, 0.5]]],
                                     lam=1.0)
    assert abs(two["estimate"] + 2 * math.log(2.0)) < 1e-12
    with pytest.raises(ValueError):
        kamath_ch6_ear_entropy_reg([[[0.5, 0.4]]])
    with pytest.raises(ValueError):
        kamath_ch6_ear_entropy_reg([[[0.5, 0.5]]], L=3)


def test_km098_is_zero_for_equal_probabilities_and_antisymmetric():
    a = kamath_ch6_log_prob_ratio_attr([0.2, 0.4], [0.1, 0.8], 2)
    b = kamath_ch6_log_prob_ratio_attr([0.1, 0.8], [0.2, 0.4], 2)
    hand = (math.log(2.0) + math.log(0.5)) / 2
    assert abs(a["estimate"] - hand) < TOL and abs(a["estimate"] + b["estimate"]) < TOL
    assert kamath_ch6_log_prob_ratio_attr([0.3], [0.3])["estimate"] == 0.0


def test_km098_checks_k_against_the_data():
    with pytest.raises(ValueError):
        kamath_ch6_log_prob_ratio_attr([0.2, 0.4], [0.1, 0.8], 3)
    with pytest.raises(ValueError):
        kamath_ch6_log_prob_ratio_attr([0.2], [0.1, 0.8])


def test_km099_returns_the_worst_generation():
    scores = [0.1, 0.9, 0.4, 0.85]
    out = kamath_ch6_emt_metric(["a", "b", "c", "d"], scores)
    assert out["estimate"] == 0.9 and out["argmax"] == "b"
    # a mean-of-inputs stub would give 0.5625
    assert abs(out["estimate"] - float(np.mean(scores))) > 0.3


def test_km099_rejects_scores_outside_the_unit_interval():
    with pytest.raises(ValueError):
        kamath_ch6_emt_metric(["a"], [1.5])


def test_km100_is_the_share_of_prompts_with_any_toxic_draw():
    sc = {"a": 0.1, "b": 0.6, "c": 0.5}
    out = kamath_ch6_toxicity_probability(
        [["a", "b"], ["a", "a"], ["c", "a"], ["a", "a"]],
        lambda y: sc[y])
    assert out["estimate"] == 0.5 and out["per_draw"] == [1.0, 0.0, 1.0, 0.0]
    assert out["n_generations"] == 8


def test_km100_flat_input_is_a_single_honest_draw():
    out = kamath_ch6_toxicity_probability(["a", "b"], [0.1, 0.9])
    assert out["estimate"] == 1.0 and out["single_draw"] is True


def test_km101_thresholds_rather_than_averaging_the_scores():
    scores = [0.49, 0.5, 0.49, 0.49]
    out = kamath_ch6_toxic_fraction(["a", "b", "c", "d"], scores)
    # exactly 0.5 counts as toxic; 0.49 does not
    assert out["estimate"] == 0.25 and out["n_toxic"] == 1
    # a mean-of-inputs stub would return 0.4925
    assert abs(out["estimate"] - float(np.mean(scores))) > 0.2


def test_km101_threshold_is_configurable_and_scores_validated():
    hi = kamath_ch6_toxic_fraction(["a", "b"], [0.6, 0.4], threshold=0.7)
    assert hi["estimate"] == 0.0
    with pytest.raises(ValueError):
        kamath_ch6_toxic_fraction(["a"], [-0.1])


def test_km102_product_agrees_with_exp_of_the_log_sum():
    p = [0.9, 0.5, 0.2, 0.75]
    out = kamath_ch6_lstm_chain_rule(p)
    assert abs(out["estimate"] - math.exp(sum(math.log(v) for v in p))) < 1e-15
    assert abs(out["estimate"] - 0.0675) < 1e-15


def test_km102_rejects_zero_and_out_of_range_probabilities():
    with pytest.raises(ValueError):
        kamath_ch6_lstm_chain_rule([0.5, 0.0])
    with pytest.raises(ValueError):
        kamath_ch6_lstm_chain_rule([])


def test_km103_normalises_and_ignores_a_shared_bias_shift():
    U = [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
    h = [0.5, 1.0]
    a = kamath_ch6_lstm_softmax_word(U, None, h, [0.0, 0.0, 0.0])
    b = kamath_ch6_lstm_softmax_word(U, None, h, [7.0, 7.0, 7.0])
    assert abs(sum(a["p"]) - 1.0) < 1e-15
    assert np.allclose(a["p"], b["p"])
    z = np.array(U) @ np.array(h)
    hand = np.exp(z) / np.exp(z).sum()
    assert np.allclose(a["p"], hand)


def test_km103_accepts_a_callable_hidden_state():
    out = kamath_ch6_lstm_softmax_word([[1.0], [0.0]],
                                       lambda c: [2.0 * c], 1.0, [0.0, 0.0])
    assert abs(out["p"][0] - math.exp(2.0) / (math.exp(2.0) + 1)) < 1e-12
    with pytest.raises(ValueError):
        kamath_ch6_lstm_softmax_word([[1.0, 0.0]], None, [1.0, 0.0],
                                     [0.0, 0.0])


def test_km104_collapses_to_km103_at_beta_zero():
    U = [[1.0, 0.0], [0.0, 2.0]]
    V = [[5.0, 5.0], [-3.0, 1.0]]
    h, e, b = [0.5, 1.0], [1.0, 1.0], [0.1, -0.2]
    base = kamath_ch6_lstm_softmax_word(U, None, h, b)
    zero = kamath_ch6_affect_lm(U, V, None, None, h, e, 0.0, b)
    assert np.allclose(base["p"], zero["p"])


def test_km104_beta_shifts_mass_toward_the_affect_category():
    U = [[1.0, 0.0], [0.0, 1.0]]
    V = [[0.0, 0.0], [1.0, 0.0]]
    h, e, b = [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]
    weak = kamath_ch6_affect_lm(U, V, None, None, h, e, 0.5, b)
    strong = kamath_ch6_affect_lm(U, V, None, None, h, e, 5.0, b)
    assert strong["p"][1] > weak["p"][1]
    assert abs(sum(strong["p"]) - 1.0) < 1e-15


def test_km105_endpoints_recover_each_component():
    assert kamath_ch6_gedi_combined_loss(2.0, 8.0, 1.0)["estimate"] == 2.0
    assert kamath_ch6_gedi_combined_loss(2.0, 8.0, 0.0)["estimate"] == 8.0
    mid = kamath_ch6_gedi_combined_loss(2.0, 8.0, 0.5)["estimate"]
    assert abs(mid - 5.0) < TOL


def test_km105_rejects_a_non_convex_weight():
    with pytest.raises(ValueError):
        kamath_ch6_gedi_combined_loss(1.0, 1.0, 1.5)


def test_km106_renormalises_yes_against_no_only():
    out = kamath_ch6_self_diagnosis_prob(
        "x", "a threat", lambda p: {"Yes": 0.2, "No": 0.2, "Maybe": 0.6})
    assert abs(out["estimate"] - 0.5) < TOL
    assert abs(out["mass_on_yes_no"] - 0.4) < TOL


def test_km106_uses_a_custom_template_and_refuses_zero_mass():
    seen = {}
    def M(prompt):
        seen["p"] = prompt
        return {"Yes": 0.9, "No": 0.1}
    kamath_ch6_self_diagnosis_prob("t", "y", M,
                                   sdg=lambda x, y: f"<{x}|{y}>")
    assert seen["p"] == "<t|y>"
    with pytest.raises(ValueError):
        kamath_ch6_self_diagnosis_prob("t", "y",
                                       lambda p: {"Yes": 0.0, "No": 0.0})


def test_km107_product_matches_the_exp_of_its_log():
    p = [0.9, 0.4, 0.5]
    out = kamath_ch6_pii_likelihood(p, ["email"], ["a", "b"], 2, 3)
    assert abs(out["estimate"] - math.exp(out["log_likelihood"])) < 1e-15
    assert abs(out["estimate"] - 0.18) < 1e-15
    assert out["context_lengths"] == [2, 3, 4]


def test_km107_checks_the_declared_lengths():
    with pytest.raises(ValueError):
        kamath_ch6_pii_likelihood([0.5, 0.5], ["e"], ["a", "b"], 2, 3)
    with pytest.raises(ValueError):
        kamath_ch6_pii_likelihood([0.5], ["e"], ["a", "b"], 1, 1)


def test_km108_boundary_of_the_guarantee():
    M = lambda D: ({"o1": 0.5, "o2": 0.5} if D == "A"
                   else {"o1": 0.25, "o2": 0.75})
    out = kamath_ch6_differential_privacy(M, "A", "B", ["o1"],
                                          math.log(2.0))
    assert abs(out["epsilon_required"] - math.log(2.0)) < 1e-12
    assert out["satisfied"] is True
    tight = kamath_ch6_differential_privacy(M, "A", "B", ["o1"],
                                            math.log(2.0) - 1e-6)
    assert tight["satisfied"] is False


def test_km108_sums_the_mass_over_the_output_subset():
    M = lambda D: ({"o1": 0.4, "o2": 0.2, "o3": 0.4} if D == "A"
                   else {"o1": 0.2, "o2": 0.1, "o3": 0.7})
    out = kamath_ch6_differential_privacy(M, "A", "B", ["o1", "o2"], 5.0)
    assert abs(out["p_A"] - 0.6) < TOL and abs(out["p_B"] - 0.3) < TOL
    with pytest.raises(ValueError):
        kamath_ch6_differential_privacy(M, "A", "B", [], 1.0)


def test_km109_returns_the_worst_sequence_not_the_average():
    S = ["w1", "w2", "w3"]
    pub = {"w1": 8.0, "w2": 3.0, "w3": 10.0}
    lm = {"w1": 4.0, "w2": 3.0, "w3": 2.0}
    out = kamath_ch6_perplexity_leakage(S, pub, lm)
    assert abs(out["estimate"] - math.log(5.0)) < 1e-12
    assert out["argmax"] == "w3" and out["n_leaking"] == 2


def test_km109_accepts_callables_and_rejects_bad_perplexities():
    out = kamath_ch6_perplexity_leakage(["a"], lambda w: 6.0,
                                        lambda w: 2.0)
    assert abs(out["estimate"] - math.log(3.0)) < 1e-12
    with pytest.raises(ValueError):
        kamath_ch6_perplexity_leakage(["a"], {"a": 0.0}, {"a": 1.0})
    with pytest.raises(ValueError):
        kamath_ch6_perplexity_leakage([], {}, {})
