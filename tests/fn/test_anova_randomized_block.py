"""Tests for anova_randomized_block.anova_randomized_block."""

from morie.fn import _array_core as np

from morie.fn.anova_randomized_block import anova_randomized_block


def test_ca9e13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = anova_randomized_block(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_ca9e13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = anova_randomized_block(x)
    assert isinstance(result, dict)
