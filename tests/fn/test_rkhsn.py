"""Tests for rkhsn.rkhs_norm."""

from morie.fn import _array_core as np

from morie.fn.rkhsn import rkhs_norm


def test_rkhsn_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    alpha = 0.05
    K = rng.normal(0, 1, (40, 40))
    result = rkhs_norm(alpha, K)
    assert isinstance(result, dict)
    assert "norm" in result


def test_rkhsn_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    alpha = 0.05
    K = rng.normal(0, 1, (5, 5))
    result = rkhs_norm(alpha, K)
    assert isinstance(result, dict)
    assert "norm" in result
