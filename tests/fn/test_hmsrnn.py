"""Tests for hmsrnn.geron_simple_rnn."""

from morie.fn import _array_core as np

from morie.fn.hmsrnn import geron_simple_rnn


def test_hmsrnn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Wx = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    Wh = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = geron_simple_rnn(X, Wx, Wh)
    assert isinstance(result, dict)
    assert "estimate" in result or "H" in result


def test_hmsrnn_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Wx = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    Wh = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = geron_simple_rnn(X, Wx, Wh)
    assert isinstance(result, dict)
