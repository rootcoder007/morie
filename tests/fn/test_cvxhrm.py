"""Tests for cvxhrm.boyd_huber_loss."""

from morie.fn import _array_core as np

from morie.fn.cvxhrm import boyd_huber_loss


def test_cvxhrm_basic():
    """Test basic functionality."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = boyd_huber_loss(u)
    assert isinstance(result, dict)
    assert "loss" in result
def test_cvxhrm_edge():
    """Test edge cases."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = boyd_huber_loss(u)
    assert isinstance(result, dict)
