"""Tests for chain_rule.chain_rule."""

from morie.fn import _array_core as np

from morie.fn.chain_rule import (
    chain_rule,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chain_rule(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chain_rule(x)
    assert isinstance(result, dict)
