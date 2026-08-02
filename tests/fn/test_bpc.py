"""Tests for bpc.bits_per_character."""

from morie.fn import _array_core as np

from morie.fn.bpc import bits_per_character


def test_bpc_basic():
    """Test basic functionality."""
    log_probs = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = bits_per_character(log_probs, N)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_bpc_edge():
    """Test edge cases."""
    log_probs = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = bits_per_character(log_probs, N)
    assert isinstance(result, dict)
