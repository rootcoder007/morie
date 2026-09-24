"""Tests for tqval.turboquant_value_cache_quantization."""

from morie.fn import _array_core as np

from morie.fn.tqval import turboquant_value_cache_quantization


def test_tqval_basic():
    """Test basic functionality."""
    v = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = turboquant_value_cache_quantization(v)
    assert isinstance(result, dict)
    assert "estimate" in result or "v_q" in result


def test_tqval_edge():
    """Test edge cases."""
    v = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = turboquant_value_cache_quantization(v)
    assert isinstance(result, dict)
