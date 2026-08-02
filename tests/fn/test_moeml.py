"""moeml: sparsely-gated mixture of experts (Shazeer et al. 2017).

    g(x) = softmax(x W_g);  keep top-k, renormalise;  y = sum_k g_k E_k(x)
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.moeml import mixture_of_experts as moe


def _experts(n, d):
    """Expert i is the affine map y = x * (i+1).

    Experts are (W_i, b_i) TUPLES, not callables -- W_i has shape
    (d_in, d_out) and the expert computes x @ W_i + b_i. Scaling each by a
    distinct factor makes the routed output identify which expert ran.
    """
    return [((i + 1) * np.eye(d), np.zeros(d)) for i in range(n)]


def test_moeml_gate_is_a_distribution_over_experts():
    rng = np.random.default_rng(3201)
    r = moe(rng.standard_normal((6, 4)), W_gate=rng.standard_normal((4, 5)),
            experts=_experts(5, 4), top_k=2)
    g = np.asarray(r["gate"])
    assert np.allclose(g.sum(axis=-1), 1.0)
    assert np.all(g >= 0)


def test_moeml_only_top_k_experts_are_selected():
    rng = np.random.default_rng(3209)
    B, k = 8, 3
    r = moe(rng.standard_normal((B, 4)), W_gate=rng.standard_normal((4, 6)),
            experts=_experts(6, 4), top_k=k)
    idx = np.asarray(r["topk_idx"])
    assert idx.shape == (B, k)
    assert np.all((idx >= 0) & (idx < 6))


def test_moeml_top_k_equal_one_routes_to_a_single_expert():
    """With k = 1 the output must be exactly that expert's output -- the
    gate weight renormalises to 1."""
    x = np.array([[1.0, 2.0]])
    W = np.array([[10.0, -10.0, -10.0], [0.0, 0.0, 0.0]])  # expert 0 wins
    out = np.asarray(moe(x, W_gate=W, experts=_experts(3, 2), top_k=1)["tensor"])
    assert out == pytest.approx(x * 1.0)


def test_moeml_sparsity_is_the_point_load_is_not_uniform_under_a_peaked_gate():
    """A gate that strongly prefers one expert must load it more heavily.
    Uniform load under a peaked gate would mean the routing does nothing."""
    rng = np.random.default_rng(3217)
    # x must be positive for a weight-only gate to favour one expert
    # UNCONDITIONALLY: with x ~ N(0,1) the logit 5*sum(x) is negative for half
    # the rows, so expert 0 would win only ~48% of the time and the test would
    # be measuring the sign of x rather than the routing.
    x = np.abs(rng.standard_normal((200, 3))) + 0.5
    W = np.zeros((3, 4)); W[:, 0] = 5.0        # expert 0 heavily favoured
    load = np.asarray(moe(x, W_gate=W, experts=_experts(4, 3), top_k=1)["load"])
    assert load[0] > 0.9, f"expected concentrated load, got {load}"
    assert np.count_nonzero(load) < 4, "top_k=1 must not use every expert"


def test_moeml_output_is_the_gated_combination_of_the_chosen_experts():
    x = np.array([[1.0, 1.0]])
    W = np.array([[1.0, 1.0, -20.0], [0.0, 0.0, 0.0]])   # experts 0,1 tie
    r = moe(x, W_gate=W, experts=_experts(3, 2), top_k=2)
    # Equal logits -> equal renormalised weights of 0.5 on experts x1 and x2.
    assert np.asarray(r["tensor"]) == pytest.approx(x * (0.5 * 1 + 0.5 * 2))


def test_moeml_more_experts_used_as_k_rises():
    rng = np.random.default_rng(3221)
    x = rng.standard_normal((50, 3))
    W = rng.standard_normal((3, 6))
    used = [len(np.unique(np.asarray(
        moe(x, W_gate=W, experts=_experts(6, 3), top_k=k)["topk_idx"])))
        for k in (1, 3, 6)]
    assert used == sorted(used)
