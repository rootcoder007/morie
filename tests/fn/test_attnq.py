"""attnq: scaled dot-product attention (Vaswani et al. 2017, eq. 1).

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.attnq import scaled_dot_product_attention as attn


def test_attnq_matches_the_closed_form():
    rng = np.random.default_rng(2901)
    Q, K, V = (rng.standard_normal((4, 8)) for _ in range(3))
    r = attn(Q, K, V)
    logits = Q @ K.T / np.sqrt(8)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    w = e / e.sum(axis=-1, keepdims=True)
    assert np.asarray(r["attn"]) == pytest.approx(w)
    assert np.asarray(r["output"]) == pytest.approx(w @ V)
    assert r["d_k"] == 8


def test_attnq_weights_are_a_distribution_per_query():
    rng = np.random.default_rng(2903)
    w = np.asarray(attn(*(rng.standard_normal((5, 6)) for _ in range(3)))["attn"])
    assert np.allclose(w.sum(axis=-1), 1.0)
    assert np.all(w >= 0)


def test_attnq_output_is_a_convex_combination_of_the_values():
    """Every output row lies inside the convex hull of V's rows, because the
    weights are non-negative and sum to 1."""
    rng = np.random.default_rng(2909)
    Q = K = rng.standard_normal((5, 4))
    V = rng.standard_normal((5, 3))
    out = np.asarray(attn(Q, K, V)["output"])
    assert np.all(out <= V.max(axis=0) + 1e-9)
    assert np.all(out >= V.min(axis=0) - 1e-9)


def test_attnq_scales_by_sqrt_d_k_not_by_d_k():
    """The 1/sqrt(d_k) factor is what keeps the softmax out of saturation as
    the head dimension grows; using 1/d_k instead gives different weights."""
    rng = np.random.default_rng(2917)
    Q, K, V = (rng.standard_normal((3, 16)) * 3 for _ in range(3))
    got = np.asarray(attn(Q, K, V)["logits"])
    assert got == pytest.approx(Q @ K.T / np.sqrt(16))
    assert not np.allclose(got, Q @ K.T / 16)


def test_attnq_a_masked_position_receives_no_weight():
    """The property the whole causal-LM setup depends on.

    The mask is ADDITIVE (0 to keep, -inf to block), not boolean -- it is
    added to the logits before the softmax. Handing in a boolean array
    silently does the wrong thing: True is added as 1.0, so the positions
    meant to be blocked get MORE weight, not none. Build it with np.where.
    """
    rng = np.random.default_rng(2927)
    Q = K = rng.standard_normal((4, 4))
    V = rng.standard_normal((4, 4))
    keep = np.tril(np.ones((4, 4), dtype=bool))
    m = np.where(keep, 0.0, -np.inf)
    w = np.asarray(attn(Q, K, V, mask=m)["attn"])
    assert np.allclose(w[np.triu_indices(4, k=1)], 0.0)
    assert np.allclose(w.sum(axis=-1), 1.0)


def test_attnq_additive_mask_of_zeros_is_a_no_op():
    """A mask that blocks nothing must leave the weights untouched."""
    rng = np.random.default_rng(2931)
    Q, K, V = (rng.standard_normal((4, 5)) for _ in range(3))
    plain = np.asarray(attn(Q, K, V)["attn"])
    masked = np.asarray(attn(Q, K, V, mask=np.zeros((4, 4)))["attn"])
    assert masked == pytest.approx(plain)


def test_attnq_identical_keys_give_uniform_attention():
    """No key is distinguishable, so every position must be weighted 1/n."""
    n = 5
    Q = np.ones((2, 3))
    K = np.ones((n, 3))
    V = np.arange(n * 2.0).reshape(n, 2)
    w = np.asarray(attn(Q, K, V)["attn"])
    assert w == pytest.approx(np.full((2, n), 1 / n))


def test_attnq_rejects_mismatched_key_value_lengths():
    with pytest.raises((ValueError, IndexError)):
        attn(np.zeros((2, 4)), np.zeros((5, 4)), np.zeros((3, 4)))
