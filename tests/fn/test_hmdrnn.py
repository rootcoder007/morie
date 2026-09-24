"""Tests for hmdrnn.geron_deep_rnn."""

from morie.fn import _array_core as np

from morie.fn.hmdrnn import geron_deep_rnn


def test_hmdrnn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_deep_rnn(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "outputs" in result


def test_hmdrnn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_deep_rnn(X)
    assert isinstance(result, dict)
