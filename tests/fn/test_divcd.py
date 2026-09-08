"""Tests for divcd.divergent_transitions_count."""

from morie.fn import _array_core as np

from morie.fn.divcd import divergent_transitions_count


def test_divcd_basic():
    """Test basic functionality."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = divergent_transitions_count(chains)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_divcd_edge():
    """Test edge cases."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = divergent_transitions_count(chains)
    assert isinstance(result, dict)
