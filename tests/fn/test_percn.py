"""Tests for percn.perceptron_activation."""

from morie.fn import _array_core as np

from morie.fn.percn import perceptron_activation


def test_percn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    w = 0.5
    b = 0.5
    result = perceptron_activation(X, w, b)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_percn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    w = 0.5
    b = 0.5
    result = perceptron_activation(X, w, b)
    assert isinstance(result, dict)
