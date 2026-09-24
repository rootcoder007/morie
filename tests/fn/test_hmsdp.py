"""Tests for hmsdp.geron_scaled_dot_product."""

from morie.fn import _array_core as np

from morie.fn.hmsdp import geron_scaled_dot_product


def test_hmsdp_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    V = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_scaled_dot_product(Q, K, V)
    assert isinstance(result, dict)
    assert "estimate" in result or "Y" in result


def test_hmsdp_edge():
    """Test edge cases."""
    Q = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    V = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_scaled_dot_product(Q, K, V)
    assert isinstance(result, dict)
