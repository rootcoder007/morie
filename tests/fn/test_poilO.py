"""Tests for poilO.poisson_loss_dnn."""

from morie.fn import _array_core as np

from morie.fn.poilO import poisson_loss_dnn


def test_poilO_basic():
    """Test basic functionality."""
    Y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    Yhat = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = poisson_loss_dnn(Y, Yhat)
    assert isinstance(result, dict)
    assert "loss" in result


def test_poilO_edge():
    """Test edge cases."""
    Y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    Yhat = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = poisson_loss_dnn(Y, Yhat)
    assert isinstance(result, dict)
