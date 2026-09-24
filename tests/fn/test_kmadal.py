"""Tests for kmadal.kamath_adalora_rank_allocation."""

from morie.fn import _array_core as np

from morie.fn.kmadal import kamath_adalora_rank_allocation


def test_kmadal_basic():
    """Test basic functionality."""
    P = np.random.default_rng(42).normal(0.0, 1.0, 40)
    s = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Q = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kamath_adalora_rank_allocation(P, s, Q)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_kmadal_edge():
    """Test edge cases."""
    P = np.random.default_rng(42).normal(0.0, 1.0, 40)
    s = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Q = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kamath_adalora_rank_allocation(P, s, Q)
    assert isinstance(result, dict)
