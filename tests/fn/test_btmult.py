"""Tests for btmult.boot_multinomial_weights."""

from morie.fn import _array_core as np

from morie.fn.btmult import boot_multinomial_weights


def test_btmult_basic():
    """Test basic functionality."""
    n = 100
    result = boot_multinomial_weights(n)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_btmult_edge():
    """Test edge cases."""
    n = 100
    result = boot_multinomial_weights(n)
    assert isinstance(result, dict)
