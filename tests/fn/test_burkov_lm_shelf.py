"""Burkov Hundred-Page LM shelf: every formula checked against an
independent route.

Gradients against central finite differences; the MSE cost against the
least-squares optimum; Kneser-Ney against a full-vocabulary sum of 1;
the autodiff graph against numeric derivatives; interpolation against
the convexity it claims. A stub returning the mean of its inputs
passes none of these.

Source: Burkov (2025) *The Hundred-Page Language Models Book*.
"""

import math

from morie.fn import _array_core as np
import pytest

from morie.fn.b101 import burkov_lm_ch1_linear_function
from morie.fn.b102 import burkov_lm_ch1_squared_error
from morie.fn.b103 import burkov_lm_ch1_mse_cost
from morie.fn.b104 import burkov_lm_ch1_linear_vector
from morie.fn.b105 import burkov_lm_ch1_cosine_similarity
from morie.fn.b106 import burkov_lm_ch1_layer1_output
from morie.fn.b107 import burkov_lm_ch1_layer2_output
from morie.fn.b108 import burkov_lm_ch1_logistic_regression
from morie.fn.b109 import burkov_lm_ch1_binary_cross_entropy
from morie.fn.b111 import burkov_lm_ch1_bce_gradients
from morie.fn.b201 import burkov_lm_ch2_categorical_cross_entropy
from morie.fn.b202 import burkov_lm_ch2_lm_next_token
from morie.fn.b203 import burkov_lm_ch2_lm_shorthand
from morie.fn.bkaddk import burkov_add_k_smoothing
from morie.fn.bkbkof import burkov_ngram_backoff
from morie.fn.bkbpc import burkov_bits_per_character
from morie.fn.bkcgr import burkov_computational_graph
from morie.fn.bkdot import burkov_dot_product
from morie.fn.bkelm import burkov_elman_rnn
from morie.fn.bkintr import burkov_ngram_interpolation
from morie.fn.bkkn import burkov_kneser_ney
from morie.fn.bklap import burkov_laplace_add_one
from morie.fn.bkngr import burkov_ngram_mle
from morie.fn.bknrm import burkov_vector_norm
from morie.fn.bkrep import burkov_repetition_penalty
from morie.fn.bktf import burkov_term_frequency
from morie.fn.bktfid import burkov_tf_idf
from morie.fn.bkunit import burkov_unit_vector
from morie.fn.bkwtie import burkov_weight_tying


def lcg_stream(seed, n):
    s = seed
    out = []
    for _ in range(n):
        s = (1664525 * s + 1013904223) % 2 ** 32
        out.append((s + 0.5) / 2 ** 32)
    return out


# --------------------------------------------------------------------
# Ch 1 linear model and losses
# --------------------------------------------------------------------

def test_the_mse_cost_is_minimised_at_the_least_squares_solution():
    u = lcg_stream(1, 40)
    x = np.array(u[:20]) * 10
    y = 2.5 * x - 1.0 + (np.array(u[20:]) - 0.5)
    A = np.vstack([x, np.ones_like(x)]).T
    w_star, b_star = np.linalg.lstsq(A, y, rcond=None)[0]
    j_star = burkov_lm_ch1_mse_cost(w_star, b_star, x, y)["cost"]
    for dw, db in [(0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05)]:
        assert burkov_lm_ch1_mse_cost(w_star + dw, b_star + db,
                                      x, y)["cost"] > j_star


def test_the_cost_decomposes_into_squared_errors_of_predictions():
    x = [1.0, 2.0, 3.0]
    y = [1.5, 3.0, 5.0]
    pred = burkov_lm_ch1_linear_function(x, 1.5, 0.0)["predictions"]
    errs = burkov_lm_ch1_squared_error(pred, y)["errors"]
    cost = burkov_lm_ch1_mse_cost(1.5, 0.0, x, y)["cost"]
    assert cost == pytest.approx(sum(errs) / 3)


def test_the_vector_form_agrees_with_the_scalar_form_in_1d():
    assert burkov_lm_ch1_linear_vector([2.0], [3.0], 1.0)["estimate"] == \
        burkov_lm_ch1_linear_function([3.0], 2.0, 1.0)["predictions"][0]


def test_mse_refuses_a_wrong_N():
    with pytest.raises(ValueError, match="dataset size"):
        burkov_lm_ch1_mse_cost(1.0, 0.0, [1.0, 2.0], [1.0, 2.0], N=5)


def test_cosine_similarity_properties():
    out = burkov_lm_ch1_cosine_similarity([1, 2, 3], [1, 2, 3])
    assert out["estimate"] == pytest.approx(1.0)
    assert burkov_lm_ch1_cosine_similarity(
        [1, 0], [-1, 0])["estimate"] == pytest.approx(-1.0)
    # scale invariance
    a, b = [1.0, 2.0, -1.0], [0.5, -1.0, 2.0]
    c1 = burkov_lm_ch1_cosine_similarity(a, b)["estimate"]
    c2 = burkov_lm_ch1_cosine_similarity(
        [7 * v for v in a], [0.1 * v for v in b])["estimate"]
    assert c1 == pytest.approx(c2)
    with pytest.raises(ValueError, match="zero vector"):
        burkov_lm_ch1_cosine_similarity([0.0, 0.0], [1.0, 1.0])


def test_the_two_layer_network_composes():
    W1 = [[1.0, -1.0], [0.5, 0.5]]
    b1 = [0.1, -0.1]
    y1 = burkov_lm_ch1_layer1_output(W1, [2.0, 1.0], b1, phi="tanh")["output"]
    y2 = burkov_lm_ch1_layer2_output([1.0, 2.0], y1, 0.3,
                                     phi="sigmoid")["estimate"]
    z = 1.0 * y1[0] + 2.0 * y1[1] + 0.3
    assert y2 == pytest.approx(1 / (1 + math.exp(-z)))


def test_relu_actually_clips():
    out = burkov_lm_ch1_layer1_output([[1.0], [-1.0]], [2.0],
                                      [0.0, 0.0], phi="relu")
    assert out["output"] == [2.0, 0.0]
    assert out["preactivation"] == [2.0, -2.0]


def test_logistic_regression_is_sigma_of_the_logit():
    out = burkov_lm_ch1_logistic_regression([1.0, -2.0], [3.0, 1.0], 0.5)
    z = 1.0 * 3.0 - 2.0 * 1.0 + 0.5
    assert out["logit"] == pytest.approx(z)
    assert out["estimate"] == pytest.approx(1 / (1 + math.exp(-z)))
    assert out["predicted_class"] == 1


def test_bce_is_zero_only_for_a_perfect_confident_prediction():
    assert burkov_lm_ch1_binary_cross_entropy(1.0, 1.0)["estimate"] == 0.0
    assert burkov_lm_ch1_binary_cross_entropy(0.0, 0.0)["estimate"] == 0.0
    assert burkov_lm_ch1_binary_cross_entropy(0.0, 1.0)["estimate"] == \
        float("inf")
    with pytest.raises(ValueError, match="targets"):
        burkov_lm_ch1_binary_cross_entropy(0.5, 0.7)


def test_the_closed_form_bce_gradient_matches_finite_differences():
    u = lcg_stream(2, 60)
    X = np.array(u[:40]).reshape(20, 2) * 4 - 2
    y = np.array([1.0 if v > 0.5 else 0.0 for v in u[40:]])
    w = np.array([0.3, -0.7]); b = 0.2

    def mean_bce(wv, bv):
        p = 1 / (1 + np.exp(-(X @ wv + bv)))
        return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

    p = 1 / (1 + np.exp(-(X @ w + b)))
    out = burkov_lm_ch1_bce_gradients(p, y, X)
    h = 1e-6
    for j in range(2):
        e = np.zeros(2); e[j] = h
        num = (mean_bce(w + e, b) - mean_bce(w - e, b)) / (2 * h)
        assert out["grad_w"][j] == pytest.approx(num, abs=1e-6)
    numb = (mean_bce(w, b + h) - mean_bce(w, b - h)) / (2 * h)
    assert out["grad_b"] == pytest.approx(numb, abs=1e-6)


# --------------------------------------------------------------------
# Ch 2 language modelling
# --------------------------------------------------------------------

def test_categorical_ce_reduces_to_neg_log_prob_of_the_true_class():
    p = [0.1, 0.6, 0.3]
    out = burkov_lm_ch2_categorical_cross_entropy(p, 1)
    assert out["estimate"] == pytest.approx(-math.log(0.6))
    with pytest.raises(ValueError, match="probability distribution"):
        burkov_lm_ch2_categorical_cross_entropy([1.0, 1.0], 0)
    with pytest.raises(ValueError, match="out of range"):
        burkov_lm_ch2_categorical_cross_entropy(p, 3)


def test_the_next_token_distribution_sums_to_one_and_counts_right():
    s = ["a", "b", "a", "c", "a", "b"]
    out = burkov_lm_ch2_lm_next_token("b", s)
    # context is "b"; successors of "b" within s: ["a"], so P(b|b) = 0
    assert out["context"] == "b"
    assert out["estimate"] == 0.0
    assert sum(out["distribution"].values()) == pytest.approx(1.0)
    out2 = burkov_lm_ch2_lm_next_token("b", ["a", "b", "a", "c", "a"])
    # successors of "a": b, c -> P(b|a) = 1/2
    assert out2["estimate"] == 0.5


def test_the_shorthand_notations_agree():
    assert burkov_lm_ch2_lm_shorthand(
        "b", ["a", "b", "a"])["notations_agree"] is True


def test_ngram_mle_and_its_refusals():
    assert burkov_ngram_mle(3, 4)["estimate"] == 0.75
    with pytest.raises(ValueError, match="undefined"):
        burkov_ngram_mle(0, 0)
    with pytest.raises(ValueError, match="impossible"):
        burkov_ngram_mle(5, 4)


def test_smoothed_distributions_sum_to_one_over_the_vocabulary():
    # 3-word vocab; prefix seen 10 times; counts 7, 3, 0
    counts = [7, 3, 0]
    V = 3
    lap = [burkov_laplace_add_one(c, 10, V)["estimate"] for c in counts]
    assert sum(lap) == pytest.approx(1.0)
    addk = [burkov_add_k_smoothing(c, 10, V, k=0.25)["estimate"]
            for c in counts]
    assert sum(addk) == pytest.approx(1.0)
    # smoothing moves mass toward the unseen word, never past the MLE
    assert lap[2] > 0
    assert lap[0] < 0.7


def test_add_1_is_add_k_at_k_equals_1():
    assert burkov_add_k_smoothing(2, 9, 5, k=1.0)["estimate"] == \
        burkov_laplace_add_one(2, 9, 5)["estimate"]


def test_interpolation_is_a_convex_combination():
    out = burkov_ngram_interpolation([0.9, 0.1, 0.5], [0.6, 0.3, 0.1])
    assert out["estimate"] == pytest.approx(0.9 * 0.6 + 0.1 * 0.3
                                            + 0.5 * 0.1)
    assert min(0.9, 0.1, 0.5) <= out["estimate"] <= max(0.9, 0.1, 0.5)
    with pytest.raises(ValueError, match="sum to 1"):
        burkov_ngram_interpolation([0.5, 0.5], [0.9, 0.3])


def test_backoff_uses_the_highest_positive_order_with_discounting():
    assert burkov_ngram_backoff([(2, 4), (9, 10)])["order_used"] == 0
    out = burkov_ngram_backoff([(0, 5), (0, 8), (3, 10)], alpha=0.5)
    assert out["order_used"] == 2
    assert out["estimate"] == pytest.approx(0.5 * 0.5 * 0.3)
    with pytest.raises(ValueError, match="nowhere left"):
        burkov_ngram_backoff([(0, 5), (0, 8)])


def test_kneser_ney_sums_to_one_over_a_full_vocabulary():
    # bigram corpus: prefix "the" seen 10 times, followed by cat x6,
    # dog x3, fish x1. Continuation counts over 8 bigram types:
    # cat appears after 3 distinct words, dog after 4, fish after 1.
    prefix = 10
    follows = {"cat": 6, "dog": 3, "fish": 1}
    cont = {"cat": 3, "dog": 4, "fish": 1}
    total_types = 8
    n_after = len(follows)
    d = 0.75
    total = sum(
        burkov_kneser_ney(follows[w], prefix,
                          (n_after, cont[w], total_types), d)["estimate"]
        for w in follows)
    # P_continuation over the words that follow this prefix must be
    # normalised for the identity to close over THIS vocabulary
    cont_mass = sum(cont.values()) / total_types
    discount_mass = sum(max(c - d, 0) for c in follows.values()) / prefix
    lam = d * n_after / prefix
    assert total == pytest.approx(discount_mass + lam * cont_mass)
    # and with the standard normalisation the whole thing is 1 when
    # continuation counts are themselves a distribution
    total_norm = sum(
        burkov_kneser_ney(
            follows[w], prefix,
            (n_after, cont[w] / sum(cont.values()) * total_types,
             total_types), d)["estimate"]
        for w in follows)
    assert total_norm == pytest.approx(1.0)


def test_bits_per_character_worked_example():
    # 2 nats/token, 100 tokens over 400 characters
    out = burkov_bits_per_character(2.0, 100, 400)
    assert out["estimate"] == pytest.approx(2.0 * 100 / (math.log(2) * 400))
    assert out["bits_per_token"] == pytest.approx(2.0 / math.log(2))


# --------------------------------------------------------------------
# Vector utilities
# --------------------------------------------------------------------

def test_dot_norm_unit_and_cosine_cohere():
    a = [3.0, 4.0]; b = [4.0, 3.0]
    dot = burkov_dot_product(a, b)["estimate"]
    na = burkov_vector_norm(a)["estimate"]
    nb = burkov_vector_norm(b)["estimate"]
    cos = burkov_lm_ch1_cosine_similarity(a, b)["estimate"]
    assert cos == pytest.approx(dot / (na * nb))
    ua = burkov_unit_vector(a)["unit"]
    assert burkov_vector_norm(ua)["estimate"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="zero vector"):
        burkov_unit_vector([0.0, 0.0])


# --------------------------------------------------------------------
# TF-IDF
# --------------------------------------------------------------------

def test_tf_idf_zeroes_a_word_in_every_document():
    corpus = [["a", "b"], ["b", "c"], ["b", "d"]]
    assert burkov_tf_idf("b", ["a", "b"], corpus)["estimate"] == \
        pytest.approx(0.0)
    out = burkov_tf_idf("a", ["a", "a", "b"], corpus)
    assert out["tf"] == 2
    assert out["estimate"] == pytest.approx(2 * math.log(3))
    with pytest.raises(ValueError, match="no corpus document"):
        burkov_tf_idf("z", ["z"], corpus)


def test_term_frequency_counts_and_normalises():
    assert burkov_term_frequency("x", ["x", "y", "x", "x"])["estimate"] == 3.0
    assert burkov_term_frequency("x", ["x", "y", "x", "x"],
                                 normalise=True)["estimate"] == \
        pytest.approx(0.75)


# --------------------------------------------------------------------
# Decoding, tying, RNN, autodiff
# --------------------------------------------------------------------

def test_repetition_penalty_lowers_odds_against_unpenalised_tokens():
    # softmax renormalisation means a penalised token's ABSOLUTE
    # probability can rise when another penalised token falls further
    # (measured here: token 1 gains 0.0003 while token 0 loses).
    # The true invariant is the odds against every UNPENALISED token.
    logits = [2.0, -1.0, 0.5]
    out = burkov_repetition_penalty(logits, [0, 1], penalty=1.5)
    z = np.array(out["penalised"])
    p_before = np.exp(logits) / np.sum(np.exp(logits))
    p_after = np.exp(z) / np.sum(np.exp(z))
    for t in (0, 1):
        assert p_after[t] / p_after[2] < p_before[t] / p_before[2]
    assert p_after[2] > p_before[2]
    # the sign split is the point: dividing a NEGATIVE logit would
    # raise its odds instead of lowering them
    assert out["penalised"][1] == pytest.approx(-1.5)


def test_a_single_penalised_token_does_lose_absolute_probability():
    logits = [2.0, -1.0, 0.5]
    out = burkov_repetition_penalty(logits, [0], penalty=1.5)
    z = np.array(out["penalised"])
    p_before = np.exp(logits) / np.sum(np.exp(logits))
    p_after = np.exp(z) / np.sum(np.exp(z))
    assert p_after[0] < p_before[0]


def test_weight_tying_matches_explicit_matrix_product():
    E = [[1.0, 2.0], [3.0, -1.0], [0.0, 1.0]]
    h = [0.5, -0.5]
    out = burkov_weight_tying(h, E)
    assert out["logits"] == pytest.approx(list(np.array(E) @ np.array(h)))
    assert out["vocab_size"] == 3
    with pytest.raises(ValueError, match="columns to"):
        burkov_weight_tying([1.0, 2.0, 3.0], E)


def test_the_elman_step_matches_a_hand_computation():
    out = burkov_elman_rnn([1.0, 0.5], [0.2, -0.1],
                           [[0.5, 0.0], [0.0, 0.5]],
                           [[1.0, 0.0], [0.0, 1.0]],
                           [[1.0, 1.0]], [0.0, 0.0], [0.1])
    h_expected = np.tanh(np.array([0.5 * 0.2 + 1.0, 0.5 * -0.1 + 0.5]))
    assert out["h"] == pytest.approx(list(h_expected))
    assert out["y"][0] == pytest.approx(float(h_expected.sum() + 0.1))


def test_autodiff_gradients_match_central_differences():
    g = [
        {"name": "wx", "op": "mul", "args": ["w", "x"]},
        {"name": "z", "op": "add", "args": ["wx", "b"]},
        {"name": "a", "op": "sigmoid", "args": ["z"]},
        {"name": "d", "op": "sub", "args": ["a", "y"]},
        {"name": "loss", "op": "square", "args": ["d"]},
    ]
    inputs = {"w": 0.7, "x": 1.3, "b": -0.2, "y": 1.0}
    out = burkov_computational_graph(g, inputs)

    def f(**kw):
        return burkov_computational_graph(g, kw)["output"]

    h = 1e-6
    for k in inputs:
        up = dict(inputs); up[k] += h
        dn = dict(inputs); dn[k] -= h
        num = (f(**up) - f(**dn)) / (2 * h)
        assert out["gradients"][k] == pytest.approx(num, abs=1e-6)


def test_autodiff_fan_out_accumulates():
    # x used twice: y = x*x + x -> dy/dx = 2x + 1
    g = [{"name": "sq", "op": "mul", "args": ["x", "x"]},
         {"name": "out", "op": "add", "args": ["sq", "x"]}]
    out = burkov_computational_graph(g, {"x": 3.0})
    assert out["output"] == 12.0
    assert out["gradients"]["x"] == pytest.approx(7.0)


def test_autodiff_rejects_a_bad_graph():
    with pytest.raises(ValueError, match="topologically"):
        burkov_computational_graph(
            [{"name": "a", "op": "add", "args": ["a2", "x"]}], {"x": 1.0})
    with pytest.raises(ValueError, match="unknown op"):
        burkov_computational_graph(
            [{"name": "a", "op": "pow", "args": ["x", "x"]}], {"x": 1.0})
