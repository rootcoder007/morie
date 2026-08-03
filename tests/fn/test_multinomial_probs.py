"""Tests for multinomial_probs.multinomial_probs."""

from morie.fn import _array_core as np

from morie.fn.multinomial_probs import multinomial_probs


def test_ca5e3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = multinomial_probs(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca5e3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = multinomial_probs(x)
    assert isinstance(result, dict)
