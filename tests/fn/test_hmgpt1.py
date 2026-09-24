"""Tests for hmgpt1.geron_gpt1."""

from morie.fn import _array_core as np

from morie.fn.hmgpt1 import geron_gpt1


def test_hmgpt1_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_gpt1(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "total_params" in result


def test_hmgpt1_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_gpt1(X)
    assert isinstance(result, dict)
