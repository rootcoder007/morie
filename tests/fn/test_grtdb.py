"""Tests for grtdb.geron_transformer_decoder_block."""

from morie.fn import _array_core as np

from morie.fn.grtdb import geron_transformer_decoder_block


def test_grtdb_basic():
    """Test basic functionality with random inputs."""
    rng = np.random.default_rng(42)
    T, S, d_model = 4, 3, 2
    x = rng.normal(0, 1, (T, d_model))
    encoder_output = rng.normal(0, 1, (S, d_model))
    I = [[1.0, 0.0], [0.0, 1.0]]
    att = {"WQ": [I], "WK": [I], "WV": [I], "WO": I}
    weights = {
        "self": att,
        "cross": att,
        "ffn": {"W1": [[0.0], [0.0]], "W2": [[0.0, 0.0]]},
    }
    result = geron_transformer_decoder_block(x, encoder_output, weights)
    assert isinstance(result, dict)
    for key in (
        "output",
        "causal_mask",
        "self_attention_weights",
        "cross_attention_weights",
        "h1",
        "h2",
        "estimate",
        "n",
        "method",
    ):
        assert key in result
    # causal_mask is a T x T lower-triangular boolean mask
    assert len(result["causal_mask"]) == T
    assert len(result["causal_mask"][0]) == T
    # output has shape (T, d_model)
    assert len(result["output"]) == T
    assert len(result["output"][0]) == d_model


def test_grtdb_edge():
    """Test edge cases (identity projections, zero ffn) matching the docstring example."""
    I = [[1.0, 0.0], [0.0, 1.0]]
    att = {"WQ": [I], "WK": [I], "WV": [I], "WO": I}
    W = {
        "self": att,
        "cross": att,
        "ffn": {"W1": [[0.0], [0.0]], "W2": [[0.0, 0.0]]},
    }
    result = geron_transformer_decoder_block(I, I, W)
    assert isinstance(result, dict)
    assert "output" in result
    assert "causal_mask" in result
    assert "self_attention_weights" in result
    assert "estimate" in result
    assert "method" in result
    assert result["causal_mask"] == [[True, False], [True, True]]
    # Token 0 attends only to itself
    assert result["self_attention_weights"][0][0] == [1.0, 0.0]
