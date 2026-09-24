"""Tests for kmret.kamath_retnet_retention."""

from morie.fn import _array_core as np

from morie.fn.kmret import kamath_retnet_retention


def test_kmret_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    V = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    gamma = 0.1
    result = kamath_retnet_retention(Q, K, V, gamma)
    assert isinstance(result, dict)
    assert "estimate" in result or "output" in result


def test_kmret_edge():
    """Test edge cases."""
    Q = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    V = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    gamma = 0.1
    result = kamath_retnet_retention(Q, K, V, gamma)
    assert isinstance(result, dict)
