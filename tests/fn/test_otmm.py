"""Tests for otmm.ot_minibatch_loss."""

from morie.fn import _array_core as np

from morie.fn.otmm import ot_minibatch_loss


def test_otmm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    batch_size = 5
    n_batches = 5
    epsilon = 0.1
    result = ot_minibatch_loss(X, Y, batch_size, n_batches, epsilon)
    assert isinstance(result, dict)
    assert "loss" in result


def test_otmm_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    batch_size = 5
    n_batches = 5
    epsilon = 0.1
    result = ot_minibatch_loss(X, Y, batch_size, n_batches, epsilon)
    assert isinstance(result, dict)
