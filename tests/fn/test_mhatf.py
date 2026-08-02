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
    A = np.asarray(r["heads"], dtype=float)
    assert A.shape == (heads, seq_len, seq_len)
    np.testing.assert_allclose(A.sum(axis=-1), 1.0, atol=1e-10)
    assert np.all(A >= 0)


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
    np.testing.assert_allclose(out, np.tile(x.mean(axis=0), (seq_len, 1)), atol=1e-10)


def test_mhatf_is_reproducible_and_validates_head_count():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 8))
    a = np.asarray(multi_head_attention_full(x, num_heads=2, seed=5)["output"], dtype=float)
    b = np.asarray(multi_head_attention_full(x, num_heads=2, seed=5)["output"], dtype=float)
    np.testing.assert_allclose(a, b, atol=1e-12)
    with pytest.raises(ValueError, match="divisible"):
        multi_head_attention_full(x, num_heads=3)
