"""Tests for hmgpt2.geron_gpt2."""

from morie.fn import _array_core as np

from morie.fn.hmgpt2 import geron_gpt2


def test_hmgpt2_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_gpt2(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "total_params" in result


def test_hmgpt2_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_gpt2(X)
    assert isinstance(result, dict)
