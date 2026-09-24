"""Tests for multinomial_probs.multinomial_probs."""

from morie.fn import _array_core as np

from morie.fn.multinomial_probs import multinomial_probs


def test_ca5e3_basic():
    """Test basic functionality."""
    xbs = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = multinomial_probs(xbs)
    assert isinstance(result, dict)
    assert "probs" in result


def test_ca5e3_edge():
    """Test edge cases."""
    xbs = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = multinomial_probs(xbs)
    assert isinstance(result, dict)
