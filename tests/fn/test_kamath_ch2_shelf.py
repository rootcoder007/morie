"""Kamath Ch 2 shelf: encoder-decoder, attention, pretraining losses,
GPT objectives and MoE, each checked by an independent route.

Attention pieces against the shared attsdp core and hand-computed
softmax; the loss family against per-position hand sums and the
inequality structure it claims; the book's Eq 2.19 mask-inside-the-
scaling convention pinned AGAINST Vaswani's mask-outside; MoE gating
against exact sparsity counts and the identity-expert isolation.

Source: Kamath, Keenan, Somers and Sorenson (2024), Ch 2
(PDF-verified: Eq 2.1-2.3 at printed p. 30).
"""

import math

from morie.fn import _array_core as np
import pytest

from morie.fn.attsdp import scaled_dot_product_attention
from morie.fn.km001 import kamath_ch2_unidirectional_encoder_state
from morie.fn.km002 import kamath_ch2_context_vector
from morie.fn.km003 import kamath_ch2_context_simplest
from morie.fn.km004 import kamath_ch2_decoder_hidden_state
from morie.fn.km005 import kamath_ch2_decoder_token_distribution
from morie.fn.km006 import kamath_ch2_seq2seq_cross_entropy
from morie.fn.km007 import kamath_ch2_attention_score
from morie.fn.km008 import kamath_ch2_attention_softmax_weights
from morie.fn.km009 import kamath_ch2_softmax_element
from morie.fn.km010 import kamath_ch2_attention_output
from morie.fn.km011 import kamath_ch2_scaled_dot_score
from morie.fn.km012 import kamath_ch2_scaled_dot_attention
from morie.fn.km013 import kamath_ch2_positional_encoding_sin
from morie.fn.km014 import kamath_ch2_positional_encoding_cos
from morie.fn.km015 import kamath_ch2_multihead_head_i
from morie.fn.km016 import kamath_ch2_multihead_concat
from morie.fn.km017 import kamath_ch2_ffn_relu
from morie.fn.km018 import kamath_ch2_layer_norm
from morie.fn.km019 import kamath_ch2_masked_attention
from morie.fn.km020 import kamath_ch2_ssl_loss
from morie.fn.km021 import kamath_ch2_clm_loss
from morie.fn.km022 import kamath_ch2_mlm_loss
from morie.fn.km023 import kamath_ch2_rtd_loss
from morie.fn.km026 import kamath_ch2_slm_loss
from morie.fn.km027 import kamath_ch2_tlm_loss
from morie.fn.km030 import kamath_ch2_nsp_loss
from morie.fn.km031 import kamath_ch2_sop_loss
from morie.fn.km032 import kamath_ch2_seq2seq_loss
from morie.fn.km033 import kamath_ch2_dae_loss
from morie.fn.km034 import kamath_ch2_gpt_unsupervised_obj
from morie.fn.km035 import kamath_ch2_gpt_supervised_softmax
from morie.fn.km036 import kamath_ch2_gpt_supervised_obj
from morie.fn.km037 import kamath_ch2_gpt_combined_obj
from morie.fn.km038 import kamath_ch2_gpt2_task_conditioning
from morie.fn.km039 import kamath_ch2_moe_output
from morie.fn.km040 import kamath_ch2_moe_topk_gating
from morie.fn.km041 import kamath_ch2_mixtral_swiglu_moe


# --------------------------------------------------------------------
# Encoder-decoder scaffolding (Eq 2.1-2.6)
# --------------------------------------------------------------------

def test_the_recurrences_accept_custom_cells_and_default_to_tanh():
    out = kamath_ch2_unidirectional_encoder_state([0.5], [0.5])
    assert out["h"][0] == pytest.approx(math.tanh(1.0))
    custom = kamath_ch2_unidirectional_encoder_state(
        [1.0], [2.0], f=lambda h, x: h * x)
    assert custom["h"] == [2.0]
    dec = kamath_ch2_decoder_hidden_state([0.1], [0.2], [0.3])
    assert dec["s"][0] == pytest.approx(math.tanh(0.6))


def test_context_mappings_and_the_simplest_case_agree():
    H = [[1.0, 2.0], [3.0, 4.0]]
    assert kamath_ch2_context_vector(H, "mean")["context"] == [2.0, 3.0]
    assert kamath_ch2_context_vector(H, "last")["context"] == [3.0, 4.0]
    out = kamath_ch2_context_simplest([3.0, 4.0], all_states=H)
    assert out["agrees_with_eq22"] is True
    with pytest.raises(ValueError, match="disagree"):
        kamath_ch2_context_simplest([9.0, 9.0], all_states=H)


def test_the_decoder_distribution_sums_to_one_and_W_projects():
    out = kamath_ch2_decoder_token_distribution([0.1], [0.2], [0.3],
                                                W=[[1.0, 0.0, 0.0],
                                                   [0.0, 0.0, 5.0]])
    assert sum(out["distribution"]) == pytest.approx(1.0)
    assert out["predicted_token"] == 1


def test_seq2seq_cross_entropy_against_a_hand_sum():
    out = kamath_ch2_seq2seq_cross_entropy([0, 1],
                                           [[0.5, 0.5], [0.25, 0.75]])
    assert out["estimate"] == pytest.approx(math.log(2) + math.log(4 / 3))
    with pytest.raises(ValueError, match="distribution"):
        kamath_ch2_seq2seq_cross_entropy([0], [[0.5, 0.6]])
    with pytest.raises(ValueError, match="does not match"):
        kamath_ch2_seq2seq_cross_entropy([0], [[1.0]], U=5)


# --------------------------------------------------------------------
# Attention (Eq 2.7-2.12, 2.15-2.16, 2.19)
# --------------------------------------------------------------------

def test_the_score_softmax_output_chain_composes_to_full_attention():
    q = [1.0, 0.0]
    K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[5.0], [-5.0]]
    scores = [kamath_ch2_attention_score(q, k)["estimate"] for k in K]
    weights = kamath_ch2_attention_softmax_weights(scores)["weights"]
    o = kamath_ch2_attention_output(weights, V)["output"]
    full = kamath_ch2_scaled_dot_attention([q], K, V)["output"][0]
    assert o == pytest.approx(full)


def test_eq_29_is_an_element_of_eq_28():
    a = [0.3, -1.2, 2.0]
    full = kamath_ch2_attention_softmax_weights(a)["weights"]
    for i, ai in enumerate(a):
        assert kamath_ch2_softmax_element(ai, a)["estimate"] == \
            pytest.approx(full[i])
    with pytest.raises(ValueError, match="not one of"):
        kamath_ch2_softmax_element(99.0, a)


def test_eq_212_delegates_to_the_shared_core():
    Q = [[1.0, 0.0]]; K = [[1.0, 0.0], [0.0, 1.0]]; V = [[1.0], [0.0]]
    km = kamath_ch2_scaled_dot_attention(Q, K, V)
    core = scaled_dot_product_attention(Q, K, V)
    assert km["output"] == core["output"]
    with pytest.raises(ValueError, match="contradicts"):
        kamath_ch2_scaled_dot_attention(Q, K, V, d_k=5)


def test_scaled_dot_score_matches_the_matrix_form():
    q = [0.3, -0.7]; k = [1.1, 0.4]
    s = kamath_ch2_scaled_dot_score(q, k)["estimate"]
    assert s == pytest.approx(np.dot(q, k) / math.sqrt(2))
    with pytest.raises(ValueError, match="contradicts"):
        kamath_ch2_scaled_dot_score(q, k, d_k=7)


def test_head_and_concat_reproduce_the_w2_multi_head():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(3, 2))
    W = [rng.normal(size=(2, 2)) for _ in range(3)]
    h1 = kamath_ch2_multihead_head_i(X, X, X, W[0], W[1], W[2])["head"]
    direct = scaled_dot_product_attention(X @ W[0], X @ W[1],
                                          X @ W[2])["output"]
    assert np.allclose(h1, direct)
    Wo = rng.normal(size=(2, 2))
    out = kamath_ch2_multihead_concat([h1], Wo)["output"]
    assert np.allclose(out, np.asarray(h1) @ Wo)


def test_the_books_mask_inside_scaling_differs_from_vaswani_for_finite_masks():
    rng = np.random.default_rng(6)
    Q = rng.normal(size=(2, 2)); K = rng.normal(size=(2, 2))
    V = rng.normal(size=(2, 1))
    M = np.array([[0.0, 3.0], [0.0, 0.0]])
    book = kamath_ch2_masked_attention(Q, K, V, M)["output"]
    vasw = scaled_dot_product_attention(Q, K, V, mask=M)["output"]
    assert not np.allclose(book, vasw)   # finite mask: conventions split
    Minf = np.array([[0.0, -np.inf], [0.0, 0.0]])
    book_inf = kamath_ch2_masked_attention(Q, K, V, Minf)["attention"]
    assert book_inf[0][1] == 0.0          # -inf: conventions agree


# --------------------------------------------------------------------
# Positional encodings, FFN, LayerNorm (Eq 2.13-2.14, 2.17-2.18)
# --------------------------------------------------------------------

def test_sin_cos_pairs_satisfy_the_pythagorean_identity():
    for i in (0, 3, 17):
        for j in (0, 1):
            s = kamath_ch2_positional_encoding_sin(i, j, 6)["estimate"]
            c = kamath_ch2_positional_encoding_cos(i, j, 6)["estimate"]
            assert s * s + c * c == pytest.approx(1.0)
    with pytest.raises(ValueError, match="below d"):
        kamath_ch2_positional_encoding_sin(0, 3, 6)


def test_the_ffn_clips_then_projects():
    out = kamath_ch2_ffn_relu([[1.0, -1.0]],
                              [[1.0, 0.0], [0.0, 1.0]],
                              [[1.0], [1.0]], [0.0, 0.0], [0.5])
    # hidden = relu([1, -1]) = [1, 0]; out = 1 + 0 + 0.5
    assert out["output"] == [[1.5]]
    assert out["hidden"] == [[1.0, 0.0]]


def test_layer_norm_centres_and_scales():
    out = kamath_ch2_layer_norm([2.0, 4.0, 6.0])
    normed = np.asarray(out["normalised"])
    assert normed.mean() == pytest.approx(0.0)
    assert normed.std() == pytest.approx(1.0)
    pinned = kamath_ch2_layer_norm([2.0, 4.0], mu=3.0, sigma=1.0, g=2.0)
    assert pinned["output"] == [-2.0, 2.0]
    with pytest.raises(ValueError, match="constant vector"):
        kamath_ch2_layer_norm([5.0, 5.0])


# --------------------------------------------------------------------
# The pretraining loss family (Eq 2.20-2.33)
# --------------------------------------------------------------------

def test_the_subset_losses_score_only_their_index_set():
    p = [0.5, 1.0, 0.25, 1.0]
    mlm = kamath_ch2_mlm_loss(p, [0, 2])
    assert mlm["estimate"] == pytest.approx(
        (math.log(2) + math.log(4)) / 2)
    # positions outside the mask never influence the loss
    p2 = [0.5, 1e-9, 0.25, 1e-9]
    assert kamath_ch2_mlm_loss(p2, [0, 2])["estimate"] == \
        pytest.approx(mlm["estimate"])
    with pytest.raises(ValueError, match="duplicates"):
        kamath_ch2_mlm_loss(p, [0, 0])
    with pytest.raises(ValueError, match="empty"):
        kamath_ch2_slm_loss(p, [])


def test_clm_scores_every_position_and_dae_shares_the_form():
    p = [0.5, 0.25]
    clm = kamath_ch2_clm_loss(p)["estimate"]
    assert clm == pytest.approx((math.log(2) + math.log(4)) / 2)
    assert kamath_ch2_dae_loss(p, "noisy")["estimate"] == \
        pytest.approx(clm)


def test_the_discriminative_loss_flips_for_replaced_tokens():
    out = kamath_ch2_rtd_loss([0.9, 0.9], [1, 0])
    assert out["per_token"][0] == pytest.approx(-math.log(0.9))
    assert out["per_token"][1] == pytest.approx(-math.log(0.1))
    assert out["accuracy"] == 0.5


def test_tlm_normalises_each_side_by_its_own_mask():
    out = kamath_ch2_tlm_loss([0.5, 1.0], [0.25], [0], [0])
    assert out["source_loss"] == pytest.approx(math.log(2))
    assert out["target_loss"] == pytest.approx(math.log(4))
    assert out["estimate"] == pytest.approx(math.log(2) + math.log(4))


def test_nsp_and_sop_share_the_binary_form():
    assert kamath_ch2_nsp_loss(0.8, "y", 1)["estimate"] == \
        pytest.approx(-math.log(0.8))
    assert kamath_ch2_nsp_loss(0.8, "y", 0)["estimate"] == \
        pytest.approx(-math.log(0.2))
    assert kamath_ch2_sop_loss(0.8, "y", 1)["estimate"] == \
        pytest.approx(kamath_ch2_nsp_loss(0.8, "y", 1)["estimate"])


def test_the_span_loss_reads_only_the_span():
    p = [1e-9, 0.5, 0.25, 1e-9]
    out = kamath_ch2_seq2seq_loss(p, "x", 1, 2)
    assert out["estimate"] == pytest.approx(
        (math.log(2) + math.log(4)) / 2)
    assert out["span_length"] == 2
    with pytest.raises(ValueError, match="inside the sequence"):
        kamath_ch2_seq2seq_loss(p, "x", 2, 9)


def test_the_composite_ssl_loss_is_the_weighted_sum():
    assert kamath_ch2_ssl_loss([1.0, 2.0, 3.0])["estimate"] == 6.0
    assert kamath_ch2_ssl_loss([1.0, 2.0], [2.0, 0.5])["estimate"] == 3.0
    with pytest.raises(ValueError, match="negative"):
        kamath_ch2_ssl_loss([1.0], [-1.0])


# --------------------------------------------------------------------
# GPT objectives (Eq 2.34-2.38)
# --------------------------------------------------------------------

def test_gpt_objectives_compose():
    l1 = kamath_ch2_gpt_unsupervised_obj([0.5, 0.5])["estimate"]
    l2 = kamath_ch2_gpt_supervised_obj([0.8])["estimate"]
    l3 = kamath_ch2_gpt_combined_obj(l1, l2, 0.5)["estimate"]
    assert l3 == pytest.approx(l2 + 0.5 * l1)
    assert l1 == pytest.approx(-2 * math.log(2))
    head = kamath_ch2_gpt_supervised_softmax("d", [1.0, 0.0],
                                             [[3.0, 0.0], [0.0, 3.0]])
    assert head["predicted_class"] == 0
    assert sum(head["probabilities"]) == pytest.approx(1.0)


def test_task_conditioning_validates_the_distribution():
    out = kamath_ch2_gpt2_task_conditioning(
        "2+2?", "math", lambda i, t: {"4": 0.9, "5": 0.1})
    assert out["output"] == "4"
    assert out["prompt"] == "math: 2+2?"
    with pytest.raises(ValueError, match="sum to 1"):
        kamath_ch2_gpt2_task_conditioning(
            "x", "t", lambda i, t: {"a": 0.9, "b": 0.9})


# --------------------------------------------------------------------
# Mixture of experts (Eq 2.39-2.41)
# --------------------------------------------------------------------

def test_moe_skips_zero_weight_experts():
    boom = lambda x: (_ for _ in ()).throw(AssertionError("expert ran"))
    out = kamath_ch2_moe_output([1.0], [1.0, 0.0],
                                [lambda x: x * 3, boom])
    assert out["output"] == [3.0]
    assert out["experts_evaluated"] == 1


def test_topk_gating_zeroes_exactly_n_minus_k_and_renormalises():
    out = kamath_ch2_moe_topk_gating([1.0], [[3.0, 1.0, 2.0, 0.5]], k=2)
    w = out["weights"]
    assert sum(1 for v in w if v == 0.0) == 2
    assert sum(w) == pytest.approx(1.0)
    assert out["selected_experts"] == [0, 2]
    # renormalised over the survivors only
    z = [3.0, 2.0]
    e = [math.exp(v - 3.0) for v in z]
    assert w[0] == pytest.approx(e[0] / sum(e))


def test_mixtral_composes_gate_and_swiglu_experts():
    ident = kamath_ch2_mixtral_swiglu_moe([1.0], [[3.0, 1.0, 2.0]])
    # identity experts isolate the gate: output = sum of top-2 weights
    assert ident["output"][0] == pytest.approx(1.0)
    W1 = [[1.0]]; W3 = [[1.0]]; W2 = [[1.0]]
    out = kamath_ch2_mixtral_swiglu_moe(
        [2.0], [[3.0, 1.0]], expert_weights=[(W1, W3, W2)] * 2)
    swiglu = (2.0 / (1 + math.exp(-2.0))) * 2.0
    assert out["output"][0] == pytest.approx(swiglu)
    with pytest.raises(ValueError, match="triple per expert"):
        kamath_ch2_mixtral_swiglu_moe([1.0], [[1.0, 2.0]],
                                      expert_weights=[(W1, W3, W2)])
