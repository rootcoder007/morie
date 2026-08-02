"""Tests for gpt2.gpt_decoder."""

from morie.fn import _array_core as np

from morie.fn.gpt2 import gpt_decoder


def test_gpt2_basic():
    """Test basic functionality."""
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = gpt_decoder(tokens, model)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_gpt2_edge():
    """Test edge cases."""
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = gpt_decoder(tokens, model)
    assert isinstance(result, dict)
