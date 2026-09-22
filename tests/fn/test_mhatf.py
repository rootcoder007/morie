"""Tests for mhatf.multi_head_attention_full."""

from morie.fn import _array_core as np
import pytest

from morie.fn.mhatf import multi_head_attention_full


def test_mhatf_shapes_and_row_stochastic_attention():
    seq_len, d_model, heads = 5, 8, 2
    rng = np.random.default_rng(0)
    r = multi_head_attention_full(rng.normal(size=(seq_len, d_model)), num_heads=heads, seed=0)

    out = np.asarray(r["output"], dtype=float)
    assert out.shape == (seq_len, d_model)
    assert int(r["num_heads"]) == heads
    assert int(r["d_k"]) == d_model // heads

    # Softmax attention: every query row is a probability distribution over keys.
    # `heads` is returned as a list of (seq_len, seq_len) weight matrices, one
    # per attention head. Stack it explicitly so we get a real rank-3 array.
    head_list = list(r["heads"])
    assert len(head_list) == heads
    A = np.stack(head_list, axis=0)
    assert A.shape == (heads, seq_len, seq_len)

    # Row-stochasticity: each query row of every head sums to 1.
    np.testing.assert_allclose(A.sum(axis=-1), 1.0, atol=1e-10)

    # Non-negativity check that works on whatever array-like A is.
    A_arr = np.asarray(A, dtype=float)
    min_val = float(np.min(A_arr))
    assert min_val >= 0.0


def test_mhatf_identity_projections_average_the_values():
    """With identity projections and a constant input, every attention weight is
    1/seq_len and the output is the column mean -- a value we can write down."""
    seq_len, d_model = 4, 4
    x = np.tile(np.arange(1.0, d_model + 1), (seq_len, 1))
    eye = np.eye(d_model)
    r = multi_head_attention_full(x, num_heads=1, W_q=eye, W_k=eye, W_v=eye, W_o=eye)
    out = np.asarray(r["output"], dtype=float)
    # Every query row attends uniformly, so each output row is the same column
    # mean. Tile it so the comparison is shape-for-shape rather than broadcast.
    expected = np.tile(x.mean(axis=0), (seq_len, 1))
    np.testing.assert_allclose(out, expected, atol=1e-10)

    # Independently verify the per-head attention weights are uniform 1/seq_len.
    A_head = np.asarray(r["heads"][0], dtype=float)
    np.testing.assert_allclose(A_head, np.full((seq_len, seq_len), 1.0 / seq_len), atol=1e-10)


def test_mhatf_is_reproducible_and_validates_head_count():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 8))
    a = np.asarray(multi_head_attention_full(x, num_heads=2, seed=5)["output"], dtype=float)
    b = np.asarray(multi_head_attention_full(x, num_heads=2, seed=5)["output"], dtype=float)
    np.testing.assert_allclose(a, b, atol=1e-12)
    with pytest.raises(ValueError, match="divisible"):
        multi_head_attention_full(x, num_heads=3)
