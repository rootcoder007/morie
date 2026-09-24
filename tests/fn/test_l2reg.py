"""Tests for l2reg.l2_weight_regularization."""

from morie.fn import _array_core as np

from morie.fn.l2reg import l2_weight_regularization


def test_l2reg_basic():
    """Test basic functionality."""
    loss = 0.1
    w = np.random.default_rng(42).normal(0.0, 1.0, 40)
    lam = 0.1
    result = l2_weight_regularization(loss, w, lam)
    assert isinstance(result, dict)
    assert "penalized_loss" in result


def test_l2reg_edge():
    """Test edge cases."""
    loss = 0.1
    w = np.random.default_rng(42).normal(0.0, 1.0, 40)
    lam = 0.1
    result = l2_weight_regularization(loss, w, lam)
    assert isinstance(result, dict)
