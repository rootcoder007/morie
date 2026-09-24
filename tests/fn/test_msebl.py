"""Tests for msebl.mse_loss_continuous."""

from morie.fn import _array_core as np

from morie.fn.msebl import mse_loss_continuous


def test_msebl_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Yhat = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mse_loss_continuous(Y, Yhat)
    assert isinstance(result, dict)
    assert "loss" in result


def test_msebl_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Yhat = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mse_loss_continuous(Y, Yhat)
    assert isinstance(result, dict)
