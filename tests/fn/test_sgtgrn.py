"""Tests for sgtgrn.sgt_graph_neural_propagation."""

from morie.fn import _array_core as np

from morie.fn.sgtgrn import sgt_graph_neural_propagation


def test_sgtgrn_basic():
    """Test basic functionality."""
    A_hat = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    X = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    W = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_graph_neural_propagation(A_hat, X, W)
    assert isinstance(result, dict)
    assert "estimate" in result or "X_next" in result


def test_sgtgrn_edge():
    """Test edge cases."""
    A_hat = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    X = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    W = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_graph_neural_propagation(A_hat, X, W)
    assert isinstance(result, dict)
