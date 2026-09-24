"""Tests for rkhsb.rkhs_bayesian_kernel."""

from morie.fn import _array_core as np

from morie.fn.rkhsb import rkhs_bayesian_kernel


def test_rkhsb_basic():
    """Test basic functionality."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    K = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = rkhs_bayesian_kernel(y, K)
    assert isinstance(result, dict)
    assert "u" in result


def test_rkhsb_edge():
    """Test edge cases."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    K = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = rkhs_bayesian_kernel(y, K)
    assert isinstance(result, dict)
