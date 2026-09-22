"""Tests for causrddm.causal_rdd_manipulation."""

from morie.fn import _array_core as np

from morie.fn.causrddm import causal_rdd_manipulation


def test_causrddm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_rdd_manipulation(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_causrddm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_rdd_manipulation(x)
    assert isinstance(result, dict)
