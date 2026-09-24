"""Tests for kmlb.kamath_moe_load_balance_loss."""

from morie.fn import _array_core as np

from morie.fn.kmlb import kamath_moe_load_balance_loss


def test_kmlb_basic():
    """Test basic functionality."""
    fractions = 1
    gate_means = 1
    N = 1
    alpha = 0.5
    result = kamath_moe_load_balance_loss(fractions, gate_means, N, alpha)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_kmlb_edge():
    """Test edge cases."""
    fractions = 1
    gate_means = 1
    N = 1
    alpha = 0.5
    result = kamath_moe_load_balance_loss(fractions, gate_means, N, alpha)
    assert isinstance(result, dict)
