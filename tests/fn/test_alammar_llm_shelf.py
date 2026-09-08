"""Alammar/Grootendorst + Vaswani shelf: every module checked by an
independent route.

Attention against hand-computed softmax rows and the recomputed-from-
scratch KV cache; losses against manual log-sum-exp; the sampler
against its own probabilities over thousands of LCG draws; chunking by
reassembly; the tokeniser by an invariant fuzz; clustering on blobs
with a planted outlier; the agent loop through recovery and
exhaustion. A stub returning the mean of its inputs passes none of
these.

Sources: Alammar and Grootendorst (2024) *Hands-On Large Language
Models*; Vaswani et al. (2017); plus the per-module papers cited in
each docstring.
"""

import math

from morie.fn import _array_core as np
import pytest

from morie.fn.alann import alammar_approximate_nearest_neighbor
from morie.fn.alaug import alammar_augmented_sbert
from morie.fn.albio import alammar_bio_tagging
from morie.fn.albow import alammar_bag_of_words
from morie.fn.albtm import alammar_bertopic_pipeline
from morie.fn.alcap import alammar_image_captioning_pipeline
from morie.fn.alcbm import alammar_conversation_buffer_memory
from morie.fn.alchat import alammar_chat_template
from morie.fn.alchk import alammar_recursive_chunking
from morie.fn.alchrj import alammar_chosen_rejected_template
from morie.fn.alclsh import alammar_classification_head
from morie.fn.alcnp import alammar_chain_prompting
from morie.fn.alcont import alammar_continued_pretraining_mlm
from morie.fn.alcsl import alammar_cosine_similarity_loss
from morie.fn.alctf import alammar_c_tfidf
from morie.fn.alctxemb import alammar_contextualized_embedding
from morie.fn.aldocemb import alammar_document_embedding_pool
from morie.fn.alembc import alammar_embedding_classifier
from morie.fn.alfrz import alammar_layer_freezing
from morie.fn.algqa import alammar_grouped_query_attention
from morie.fn.algrdy import alammar_greedy_decoding
from morie.fn.alhds import alammar_hdbscan_cluster
from morie.fn.alinfn import alammar_infonce_loss
from morie.fn.alkvc2 import alammar_kv_cache_lookup
from morie.fn.alldat import alammar_lda_topic_distribution
from morie.fn.allktmpl import alammar_instruction_data_template
from morie.fn.alllmj import alammar_llm_as_judge
from morie.fn.almnrl import alammar_multiple_negatives_ranking
from morie.fn.almqa import alammar_multi_query_attention
from morie.fn.almqr import alammar_multi_query_retrieval
from morie.fn.almrr import alammar_mean_reciprocal_rank
from morie.fn.almteb import alammar_mteb_benchmark_score
from morie.fn.alndcg import alammar_ndcg_at_k
from morie.fn.alnerh import alammar_ner_token_head
from morie.fn.alnsmp import alammar_negative_sampling_skipgram
from morie.fn.alocp import alammar_openclip_contrastive
from morie.fn.alocv import alammar_output_verification
from morie.fn.alrck import alammar_recall_at_k
from morie.fn.alreact import alammar_react_agent_loop
from morie.fn.alrmt import alammar_reward_model_training_bt
from morie.fn.alsft import alammar_setfit_twostep
from morie.fn.alsmc import alammar_simcse_dropout_aug
from morie.fn.alspl import alammar_sampling_decoding
from morie.fn.alswa import alammar_sliding_window_attention
from morie.fn.alt5c import alammar_t5_text_to_text_classify
from morie.fn.altkemb import alammar_token_embedding_lookup
from morie.fn.altkp import alammar_tokenization_pipeline
from morie.fn.altrip import alammar_sbert_triplet_loss
from morie.fn.altsd import alammar_tsdae_objective
from morie.fn.alumap import alammar_umap_projection
from morie.fn.alvit import alammar_vit_patch_embedding
from morie.fn.alvocb import alammar_tokenizer_vocab_overlap
from morie.fn.alzsc import alammar_zero_shot_classification
from morie.fn.attmh import multi_head_attention
from morie.fn.attsdp import scaled_dot_product_attention


# --------------------------------------------------------------------
# Attention family
# --------------------------------------------------------------------

def test_sdp_attention_matches_a_hand_computed_softmax():
    out = scaled_dot_product_attention([[1.0, 0.0]],
                                       [[1.0, 0.0], [0.0, 1.0]],
                                       [[5.0], [-5.0]])
    s = 1.0 / math.sqrt(2)
    w = math.exp(s) / (math.exp(s) + 1.0)
    assert out["attention"][0][0] == pytest.approx(w)
    assert out["output"][0][0] == pytest.approx(5.0 * w - 5.0 * (1 - w))


def test_attention_rows_always_sum_to_one():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(4, 3)); K = rng.normal(size=(6, 3))
    V = rng.normal(size=(6, 2))
    A = np.asarray(scaled_dot_product_attention(Q, K, V)["attention"])
    assert np.allclose(A.sum(axis=1), 1.0)


def test_a_boolean_mask_zeroes_the_dropped_positions():
    m = np.array([[True, False]])
    out = scaled_dot_product_attention([[1.0, 0.0]],
                                       [[1.0, 0.0], [0.0, 1.0]],
                                       [[7.0], [0.0]], mask=m)
    assert out["attention"][0] == [1.0, 0.0]
    assert out["output"][0][0] == 7.0


def test_multi_head_with_identity_projections_reduces_to_sdp():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, 2))
    I2 = np.eye(2)
    mh = multi_head_attention(X, X, X, [I2], [I2], [I2], I2, 1)
    sdp = scaled_dot_product_attention(X, X, X)
    assert np.allclose(mh["output"], sdp["output"])


def test_multi_head_refuses_a_broadcast_projection_list():
    I2 = np.eye(2)
    with pytest.raises(ValueError, match="one per head"):
        multi_head_attention(I2, I2, I2, [I2], [I2], [I2], I2, 2)


def test_gqa_limits_recover_mqa_and_full_attention():
    rng = np.random.default_rng(2)
    Qh = [rng.normal(size=(3, 2)) for _ in range(4)]
    Ks = rng.normal(size=(3, 2)); Vs = rng.normal(size=(3, 2))
    mqa = alammar_multi_query_attention(Qh, Ks, Vs, 4)
    gqa1 = alammar_grouped_query_attention(Qh, [Ks], [Vs], 4, 1)
    assert np.allclose(mqa["output"], gqa1["output"])
    Kg = [rng.normal(size=(3, 2)) for _ in range(4)]
    Vg = [rng.normal(size=(3, 2)) for _ in range(4)]
    gqaH = alammar_grouped_query_attention(Qh, Kg, Vg, 4, 4)
    per_head = [np.asarray(scaled_dot_product_attention(
        Qh[i], Kg[i], Vg[i])["output"]) for i in range(4)]
    assert np.allclose(gqaH["output"], np.concatenate(per_head, axis=1))
    with pytest.raises(ValueError, match="divisible"):
        alammar_grouped_query_attention(Qh, [Ks], [Vs], 4, 3)


def test_sliding_window_attention_is_zero_outside_the_band():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(6, 2))
    out = alammar_sliding_window_attention(X, X, X, 2)
    A = np.asarray(out["attention"])
    for i in range(6):
        for j in range(6):
            if j > i or j < i - 1:
                assert A[i, j] == 0.0
    assert np.allclose(A.sum(axis=1), 1.0)


def test_the_kv_cache_step_equals_full_attention_recomputed():
    rng = np.random.default_rng(4)
    K = rng.normal(size=(4, 3)); V = rng.normal(size=(4, 2))
    k5 = rng.normal(size=(1, 3)); v5 = rng.normal(size=(1, 2))
    q = rng.normal(size=(1, 3))
    cached = alammar_kv_cache_lookup(K, V, k5, v5, q)
    full = scaled_dot_product_attention(q, np.vstack([K, k5]),
                                        np.vstack([V, v5]))
    assert np.allclose(cached["output"], full["output"][0])
    assert cached["cache_length"] == 5


# --------------------------------------------------------------------
# Heads and embeddings
# --------------------------------------------------------------------

def test_classification_and_ner_heads_are_softmax_linear():
    out = alammar_classification_head([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]],
                                      [0.0, 0.0])
    assert out["predicted_class"] == 1
    assert sum(out["probabilities"]) == pytest.approx(1.0)
    ner = alammar_ner_token_head([[1.0, 0.0], [0.0, 1.0]],
                                 [[2.0, 0.0], [0.0, 2.0]], [0.0, 0.0],
                                 [0, 1])
    assert ner["predicted_tags"] == [0, 1]
    assert ner["cross_entropy"] == pytest.approx(
        -math.log(math.exp(2) / (math.exp(2) + 1)))


def test_embedding_lookup_and_masked_pooling():
    assert alammar_token_embedding_lookup(
        [1, 0], [[1.0, 2.0], [3.0, 4.0]])["embeddings"] == \
        [[3.0, 4.0], [1.0, 2.0]]
    with pytest.raises(ValueError, match="vocabulary"):
        alammar_token_embedding_lookup([2], [[1.0], [2.0]])
    pool = alammar_document_embedding_pool([[2.0], [4.0], [99.0]],
                                           [1, 1, 0])
    assert pool["embedding"] == [3.0]
    with pytest.raises(ValueError, match="all-padding"):
        alammar_document_embedding_pool([[1.0]], [0])


def test_contextual_embedding_extraction_and_variation_flag():
    stack = [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 1.0], [2.0, 2.0]]]
    out = alammar_contextualized_embedding(stack, -1, 1)
    assert out["embedding"] == [2.0, 2.0]
    assert out["context_varies"] is True
    same = alammar_contextualized_embedding(stack, 0, 0)
    assert same["context_varies"] is False


def test_vit_patches_tile_exactly_and_project():
    img = np.arange(16.0).reshape(4, 4)
    E = np.eye(4)
    out = alammar_vit_patch_embedding(img, 2, E)
    assert out["n_patches"] == 4
    assert out["sequence"][0] == [0.0, 1.0, 4.0, 5.0]
    with pytest.raises(ValueError, match="tile"):
        alammar_vit_patch_embedding(np.zeros((5, 4)), 2, E)


# --------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------

def test_infonce_matches_a_manual_log_sum_exp():
    a = [1.0, 0.0]; p = [1.0, 0.0]; negs = [[0.0, 1.0], [-1.0, 0.0]]
    tau = 0.1
    out = alammar_infonce_loss(a, p, negs, tau)
    sims = [1.0, 0.0, -1.0]
    zs = [s / tau for s in sims]
    m = max(zs)
    manual = m + math.log(sum(math.exp(z - m) for z in zs)) - zs[0]
    assert out["estimate"] == pytest.approx(manual)


def test_mnr_and_simcse_share_the_in_batch_softmax_shape():
    A = [[1.0, 0.0], [0.0, 1.0]]
    out = alammar_multiple_negatives_ranking(A, A, tau=1.0)
    manual = -math.log(math.e / (math.e + 1.0))
    assert out["estimate"] == pytest.approx(manual)
    assert alammar_simcse_dropout_aug(A, A, tau=1.0)["estimate"] == \
        pytest.approx(manual)


def test_clip_loss_is_symmetric_and_minimal_on_aligned_towers():
    I = [[1.0, 0.0], [0.0, 1.0]]
    aligned = alammar_openclip_contrastive(I, I, tau=0.5)
    swapped = alammar_openclip_contrastive(I, [I[1], I[0]], tau=0.5)
    assert aligned["estimate"] < swapped["estimate"]
    assert aligned["image_to_text_loss"] == pytest.approx(
        aligned["text_to_image_loss"])


def test_triplet_and_cosine_losses_behave():
    t = alammar_sbert_triplet_loss([[0.0]], [[0.5]], [[0.6]], margin=1.0)
    assert t["losses"][0] == pytest.approx(0.5 - 0.6 + 1.0)
    assert t["active"] == [True]
    c = alammar_cosine_similarity_loss([[1.0, 0.0]], [[0.0, 1.0]], [0.0])
    assert c["estimate"] == pytest.approx(0.0)


def test_skipgram_loss_is_stable_at_extreme_scores():
    out = alammar_negative_sampling_skipgram([100.0], [1.0], [[-1.0]])
    assert np.isfinite(out["estimate"])
    assert out["estimate"] == pytest.approx(0.0, abs=1e-8)


def test_bradley_terry_loss_and_accuracy_cohere():
    out = alammar_reward_model_training_bt([2.0, 0.0], [0.0, 1.0])
    assert out["pair_accuracy"] == 0.5
    assert out["losses"][0] == pytest.approx(math.log(1 + math.exp(-2)))
    assert out["losses"][1] == pytest.approx(math.log(1 + math.exp(1)))


# --------------------------------------------------------------------
# Metrics and decoding
# --------------------------------------------------------------------

def test_retrieval_metrics_on_worked_examples():
    assert alammar_mean_reciprocal_rank([[3, 1], [9]], [[1], [7]])[
        "estimate"] == pytest.approx(0.25)
    assert alammar_recall_at_k([1, 2, 3], [2, 9], 2)["estimate"] == 0.5
    perfect = alammar_ndcg_at_k([3, 2, 1], 3)
    assert perfect["estimate"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="undefined"):
        alammar_ndcg_at_k([0, 0], 2)


def test_mteb_weighting_differs_from_the_flat_mean():
    out = alammar_mteb_benchmark_score(
        {"a": 1.0, "b": 0.0, "c": 0.5}, {"a": "x", "b": "x", "c": "y"})
    assert out["estimate"] == pytest.approx(0.5)
    assert out["flat_task_mean"] == pytest.approx(0.5)
    out2 = alammar_mteb_benchmark_score(
        {"a": 1.0, "b": 1.0, "c": 0.0}, {"a": "x", "b": "x", "c": "y"})
    assert out2["estimate"] == pytest.approx(0.5)
    assert out2["flat_task_mean"] == pytest.approx(2 / 3)
    assert out2["weighting_matters"] is True


def test_greedy_decoding_reports_ties():
    out = alammar_greedy_decoding([[1.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    assert out["tokens"] == [0, 1]
    assert out["had_ties"] == [True, False]


def test_the_sampler_tracks_its_own_probabilities():
    # over 4000 LCG draws the empirical frequency of token 0 must be
    # within 3 sd of softmax(logits)[0]
    logits = [[1.0, 0.0]] * 4000
    out = alammar_sampling_decoding(logits, seed=7)
    p0 = math.exp(1) / (math.exp(1) + 1)
    freq = out["tokens"].count(0) / 4000
    sd = math.sqrt(p0 * (1 - p0) / 4000)
    assert abs(freq - p0) < 3 * sd
    again = alammar_sampling_decoding(logits, seed=7)
    assert again["tokens"] == out["tokens"]


# --------------------------------------------------------------------
# Text and RAG utilities
# --------------------------------------------------------------------

def test_bow_ctfidf_bio_and_vocab_overlap():
    assert alammar_bag_of_words(["a", "b", "a", "z"],
                                ["a", "b", "c"])["bow_vector"] == [2, 1, 0]
    ct = alammar_c_tfidf([[4.0, 0.0], [0.0, 4.0]])
    assert ct["top_term_per_class"] == [0, 1]
    tags = alammar_bio_tagging(["a", "b", "c"], [(0, 2, "PER")],
                               scheme="BIOES")
    assert tags["tags"] == ["B-PER", "E-PER", "O"]
    with pytest.raises(ValueError, match="overlap"):
        alammar_bio_tagging(["a", "b"], [(0, 2, "X"), (1, 2, "Y")])
    assert alammar_tokenizer_vocab_overlap(["a", "b"],
                                           ["b", "c"])["estimate"] == \
        pytest.approx(1 / 3)


def test_chunking_reassembles_and_respects_the_cap():
    # target 4 forces BOTH separator tiers to fire, so every emitted
    # chunk is separator-free and reassembly is exact
    text = "aa bb. cc dd. ee ff gg. hh"
    out = alammar_recursive_chunking(text, separators=[". ", " "],
                                     target_size=4)
    assert all(len(c) <= 4 for c in out["chunks"])
    rebuilt = "".join(out["chunks"])
    assert rebuilt == text.replace(". ", "").replace(" ", "")
    ov = alammar_recursive_chunking("abcdefgh", separators=[],
                                    target_size=4, overlap=2)
    assert ov["chunks"] == ["abcd", "cdefgh"[:6]]


def test_memory_template_and_preference_records():
    m = alammar_conversation_buffer_memory([("u1", "a1"), ("u2", "a2")], 1)
    assert m["memory"] == [("u2", "a2")]
    assert m["turns_forgotten"] == 1
    t = alammar_chat_template([("user", "hi")],
                              {"user": ("<u>", "</u>")})
    assert t["prompt"] == "<u>hi</u>"
    with pytest.raises(ValueError, match="no template tokens"):
        alammar_chat_template([("robot", "x")], {"user": ("", "")})
    with pytest.raises(ValueError, match="no preference"):
        alammar_chosen_rejected_template(["p"], ["same"], ["same"])


def test_instruction_template_masks_exactly_the_output():
    out = alammar_instruction_data_template(
        [{"instruction": "add", "input": "2 2", "output": "four"}])
    s, e = out["output_spans"][0]
    assert out["texts"][0][s:e] == "four"
    assert "add" not in out["texts"][0][s:e]


def test_tokeniser_invariant_every_token_in_vocab_or_unk():
    v = ["[CLS]", "[SEP]", "[UNK]", "un", "##happy", "##ly", "dog", "run"]
    out = alammar_tokenization_pipeline("Unhappily running dogs", v)
    vs = set(v)
    assert all(t in vs for t in out["tokens"])
    assert out["tokens"][0] == "[CLS]" and out["tokens"][-1] == "[SEP]"
    happy = alammar_tokenization_pipeline("unhappy dog", v)
    assert happy["tokens"] == ["[CLS]", "un", "##happy", "dog", "[SEP]"]


# --------------------------------------------------------------------
# Clustering, topics, projection
# --------------------------------------------------------------------

def test_hdbscan_finds_planted_blobs_and_flags_the_outlier():
    X = [[0, 0], [0.1, 0], [0, 0.1], [5, 5], [5.1, 5], [5, 5.1], [20, 20]]
    out = alammar_hdbscan_cluster(X, 3, 2)
    assert out["n_clusters"] == 2
    assert out["labels"][6] == -1
    assert out["labels"][0] == out["labels"][1] == out["labels"][2]
    assert out["labels"][3] == out["labels"][4] == out["labels"][5]
    assert out["labels"][0] != out["labels"][3]


def test_umap_reduces_its_own_objective_and_separates_blobs():
    X = [[0, 0], [0.2, 0], [0, 0.2], [8, 8], [8.2, 8], [8, 8.2]]
    out = alammar_umap_projection(X, n_neighbors=2, n_steps=500,
                                  learning_rate=0.1)
    assert out["objective_decreased"] is True
    Z = np.asarray(out["embedding"])
    intra = np.linalg.norm(Z[0] - Z[1])
    inter = np.linalg.norm(Z[0] - Z[3])
    assert inter > intra


def test_lda_recovers_the_planted_topic_split():
    docs = [["cat", "dog", "cat", "dog"]] * 3 + \
        [["stock", "bond", "stock", "bond"]] * 3
    out = alammar_lda_topic_distribution(docs, 2, n_iter=300, seed=3)
    th = np.asarray(out["theta"])
    assert np.allclose(th.sum(axis=1), 1.0)
    animal_topic = int(np.argmax(th[0]))
    finance_topic = int(np.argmax(th[3]))
    assert animal_topic != finance_topic
    assert all(int(np.argmax(th[i])) == animal_topic for i in range(3))
    assert all(int(np.argmax(th[i])) == finance_topic for i in range(3, 6))


def test_bertopic_end_to_end_names_the_planted_topics():
    docs = [["cat", "dog", "cat"], ["dog", "cat"],
            ["stock", "bond", "stock"], ["bond", "stock", "bond"]]
    out = alammar_bertopic_pipeline(docs, [[0, 0], [0.1, 0.1],
                                           [5, 5], [5.1, 5.1]], 2)
    assert out["n_topics"] == 2
    words = set(out["topic_top_word"].values())
    assert words & {"cat", "dog"}
    assert words & {"stock", "bond"}


def test_the_embedding_classifier_actually_learns():
    X = [[0, 0], [0.2, 0.1], [0.1, 0.2], [5, 5], [5.2, 5.1], [5.1, 5.2]]
    y = [0, 0, 0, 1, 1, 1]
    out = alammar_embedding_classifier(X, y)
    assert out["train_accuracy"] == 1.0
    assert out["predictions"] == y


def test_ann_measures_its_own_accuracy():
    pts = [[float(i), 0.0] for i in range(6)] + [[50.0, 50.0]]
    nbrs = [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5]]
    idx = {"points": pts, "neighbors": nbrs, "entry": 0}
    out = alammar_approximate_nearest_neighbor([4.4, 0.0], idx)
    assert out["found_exact"] is True
    assert out["nearest"] == 4
    # an entry with NO outgoing edges cannot reach the truth, and
    # says so (a [[0]] * 7 neighbour list is not disconnected: node 6
    # would step straight to node 0 -- the first draft of this test
    # made exactly that mistake)
    idx2 = {"points": pts, "neighbors": [[1], [0], [1], [2], [3], [4],
                                         []], "entry": 6}
    out2 = alammar_approximate_nearest_neighbor([0.0, 0.0], idx2,
                                                ef_search=1)
    assert out2["found_exact"] is False


# --------------------------------------------------------------------
# Orchestration with caller-supplied models
# --------------------------------------------------------------------

def test_zero_shot_and_t5_classification_pick_the_right_label():
    def nli(premise, hypothesis):
        return 5.0 if "sport" in hypothesis and "goal" in premise else 0.0
    out = alammar_zero_shot_classification("a late goal won it",
                                           ["sport", "finance"], nli)
    assert out["predicted_label"] == "sport"
    def t5(inp, label):
        return 0.0 if label == "positive" and "good" in inp else -3.0
    out2 = alammar_t5_text_to_text_classify("good film",
                                            ["positive", "negative"], t5)
    assert out2["predicted_label"] == "positive"
    assert sum(out2["probabilities"].values()) == pytest.approx(1.0)


def test_judge_variance_is_surfaced_and_gates_refuse_prose():
    def judge(rubric, resp, s):
        return {"a": [1.0, 3.0], "b": [2.0, 2.0]}[resp][s]
    out = alammar_llm_as_judge(["a", "b"], "rubric", judge, n_samples=2)
    assert out["scores"] == [2.0, 2.0]
    assert out["judge_sd"][0] > out["judge_sd"][1]
    def verifier(resp, crit):
        return "PASS" if crit == "ok" else "maybe"
    with pytest.raises(ValueError, match="only 'PASS' or"):
        alammar_output_verification("x", ["ok", "hmm"], verifier)
    def strict(resp, crit):
        return "FAIL" if crit == "hard" else "PASS"
    out2 = alammar_output_verification("x", ["ok", "hard"], strict)
    assert out2["passed"] is False
    assert out2["failed_criteria"] == ["hard"]


def test_chains_retrieval_and_the_react_loop():
    chain = alammar_chain_prompting(
        "3", [lambda y, x: f"double {x}",
              lambda y, x: f"add one to {y}"],
        lambda p: str(int(p.split()[-1]) * 2) if "double" in p
        else str(int(p.split()[-1]) + 1))
    assert chain["final_output"] == "7"
    assert len(chain["steps"]) == 2

    corpus = {"q": [1, 2], "q0": [2, 3], "q1": [4]}
    out = alammar_multi_query_retrieval(
        "q", 2, lambda q: corpus[q], lambda q, i: f"{q}{i}")
    assert out["documents"] == [1, 2, 3, 4]
    assert out["added_per_query"] == [2, 1, 1]

    def model(ctx):
        last = ctx[-1]
        if "query" in last:
            return {"thought": "look up", "action": "search",
                    "action_input": "x"}
        if last.get("observation", "").startswith("ERROR"):
            return {"thought": "recover", "final": "gave up cleanly"}
        return {"thought": "answer", "final": last["observation"]}
    tools = {"search": lambda q: "42"}
    out2 = alammar_react_agent_loop("what is x", tools, model)
    assert out2["answer"] == "42"
    assert out2["exhausted"] is False
    out3 = alammar_react_agent_loop("x", {}, lambda c: {
        "thought": "t", "action": "missing"}, max_steps=2)
    assert out3["exhausted"] is True
    assert out3["answer"] is None
    assert "ERROR" in out3["trace"][0]["observation"]


def test_captioning_projects_and_enforces_dimensions():
    out = alammar_image_captioning_pipeline(
        "img", lambda im: [1.0, 2.0], [[1.0, 0.0]],
        lambda z, p: f"caption of {z}")
    assert out["projected"] == [1.0]
    with pytest.raises(ValueError, match="columns"):
        alammar_image_captioning_pipeline(
            "img", lambda im: [1.0, 2.0, 3.0], [[1.0, 0.0]],
            lambda z, p: "x")


def test_freezing_schedules_thaw_monotonically():
    out = alammar_layer_freezing(6, 3)
    masks = out["masks"]
    assert out["trainable_per_stage"] == [2, 4, 6]
    for a, b in zip(masks, masks[1:]):
        for x, y in zip(a, b):
            assert (not x) or y      # once thawed, never refrozen


def test_continued_pretraining_reports_the_loss_curve():
    out = alammar_continued_pretraining_mlm(
        ["doc"], lambda docs, step: 1.0 / (step + 1), 5)
    assert out["mlm_loss_curve"] == [1.0, 0.5, 1 / 3, 0.25, 0.2]
    assert out["mlm_improved"] is True


def test_augmented_sbert_measures_gold_agreement():
    ce = lambda a, b: 1.0 if a == b else 0.0
    out = alammar_augmented_sbert(
        [("x", "x"), ("x", "y")], ce,
        gold_pairs=[("p", "p"), ("p", "q"), ("r", "r"), ("r", "s")],
        gold_labels=[1.0, 0.0, 1.0, 0.0])
    assert out["n_silver"] == 2
    assert out["cross_encoder_gold_agreement"] == pytest.approx(1.0)


def test_setfit_builds_the_pair_sets_and_a_working_head():
    X = [[0, 0], [0.1, 0], [5, 5], [5.1, 5]]
    out = alammar_setfit_twostep(X, [0, 0, 1, 1])
    assert out["n_positive"] == 2
    assert out["n_negative"] == 4
    assert out["head_train_accuracy"] == 1.0


def test_tsdae_deletion_is_deterministic_and_the_nll_counts_all_tokens():
    toks = list("abcdefghij")
    a = alammar_tsdae_objective(toks, seed=5)
    b = alammar_tsdae_objective(toks, seed=5)
    assert a["corrupted"] == b["corrupted"]
    assert len(a["corrupted"]) + len(a["deleted"]) == 10
    out = alammar_tsdae_objective(toks, seed=5,
                                  reconstruction_logprob=[-0.1] * 10)
    assert out["loss"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="ORIGINAL"):
        alammar_tsdae_objective(toks, seed=5,
                                reconstruction_logprob=[-0.1] * 4)
