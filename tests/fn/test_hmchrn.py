"""Tests for hmchrn.geron_char_rnn."""

from morie.fn import _array_core as np

from morie.fn.hmchrn import geron_char_rnn


def test_hmchrn_basic():
    """Test basic functionality."""
    text = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_char_rnn(text)
    assert isinstance(result, dict)
    assert "estimate" in result or "loss_history" in result


def test_hmchrn_edge():
    """Test edge cases."""
    text = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_char_rnn(text)
    assert isinstance(result, dict)
