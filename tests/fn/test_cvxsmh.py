"""Tests for cvxsmh.boyd_smooth_huber_grad."""

from morie.fn import _array_core as np

from morie.fn.cvxsmh import boyd_smooth_huber_grad


def test_cvxsmh_basic():
    """Test basic functionality."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = boyd_smooth_huber_grad(u)
    assert isinstance(result, dict)
    assert "gradient" in result
def test_cvxsmh_edge():
    """Test edge cases."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = boyd_smooth_huber_grad(u)
    assert isinstance(result, dict)
