"""Tests for irtgpc.generalized_partial_credit."""

from morie.fn import _array_core as np

from morie.fn.irtgpc import generalized_partial_credit


def test_irtgpc_basic():
    """Test basic functionality."""
    y = 0.5
    theta = 0.5
    a = 0.5
    b_j = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = generalized_partial_credit(y, theta, a, b_j)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_irtgpc_edge():
    """Test edge cases."""
    y = 0.5
    theta = 0.5
    a = 0.5
    b_j = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = generalized_partial_credit(y, theta, a, b_j)
    assert isinstance(result, dict)
