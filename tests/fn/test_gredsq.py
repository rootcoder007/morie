"""Tests for gredsq.geron_encoder_decoder_seq2seq."""

from morie.fn import _array_core as np

from morie.fn.gredsq import geron_encoder_decoder_seq2seq


def test_gredsq_basic():
    """Test basic functionality."""
    encoder = lambda x: [float(sum(x))]
    decoder = lambda y_prev, c, t: [c[0] - t]
    x = [1.0, 2.0]
    max_out_len = 3
    result = geron_encoder_decoder_seq2seq(encoder, decoder, x, max_out_len)
    assert isinstance(result, dict)
    assert "outputs" in result
    assert "context" in result
    assert "n_steps" in result
    assert "stopped_early" in result
    assert "context_dim" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert result["context"] == [3.0]
    assert result["outputs"] == [[2.0], [1.0], [0.0]]
    assert result["n_steps"] == 3
    assert result["stopped_early"] is False


def test_gredsq_edge():
    """Test edge cases."""
    encoder = lambda x: [float(sum(x))]
    decoder = lambda y_prev, c, t: [c[0] - t]
    x = [1.0, 2.0]
    max_out_len = 10
    result = geron_encoder_decoder_seq2seq(encoder, decoder, x, max_out_len, eos_token=0.0)
    assert isinstance(result, dict)
    assert result["outputs"] == [[2.0], [1.0], [0.0]]
    assert result["n_steps"] == 3
    assert result["stopped_early"] is True
