"""Tests for kblup.kernel_blup."""

from morie.fn import _array_core as np

from morie.fn.kblup import kernel_blup


def test_kblup_basic():
    """Test basic functionality."""
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kernel_blup(Z, K)
    assert isinstance(result, dict)
    assert "K_star" in result


def test_kblup_edge():
    """Test edge cases."""
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kernel_blup(Z, K)
    assert isinstance(result, dict)
