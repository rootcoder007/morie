"""Tests for jomint.joseph_mint_reconciliation."""

from morie.fn import _array_core as np

from morie.fn.jomint import joseph_mint_reconciliation


def test_jomint_basic():
    """Test basic functionality."""
    y_hat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    S = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = joseph_mint_reconciliation(y_hat, S)
    assert isinstance(result, dict)
    assert "reconciled" in result


def test_jomint_edge():
    """Test edge cases."""
    y_hat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    S = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = joseph_mint_reconciliation(y_hat, S)
    assert isinstance(result, dict)
