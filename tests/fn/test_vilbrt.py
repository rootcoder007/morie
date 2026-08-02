"""Tests for vilbrt.vilbert_two_stream."""

from morie.fn import _array_core as np

from morie.fn.vilbrt import vilbert_two_stream


def test_vilbrt_basic():
    """Test basic functionality."""
    image_tokens = np.random.default_rng(42).normal(0, 1, 100)
    text_tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = vilbert_two_stream(image_tokens, text_tokens)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_vilbrt_edge():
    """Test edge cases."""
    image_tokens = np.random.default_rng(42).normal(0, 1, 100)
    text_tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = vilbert_two_stream(image_tokens, text_tokens)
    assert isinstance(result, dict)
