"""Tests for topkS.top_k_sampling."""

from morie.fn import _array_core as np

from morie.fn.topkS import top_k_sampling


def test_topkS_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0.0, 1.0, 40)
    k = 5
    temp = 0.1
    result = top_k_sampling(logits, k, temp)
    assert isinstance(result, dict)
    assert "tensor" in result


def test_topkS_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0.0, 1.0, 40)
    k = 5
    temp = 0.1
    result = top_k_sampling(logits, k, temp)
    assert isinstance(result, dict)
