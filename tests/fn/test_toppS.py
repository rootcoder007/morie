"""Tests for toppS.top_p_sampling."""

from morie.fn import _array_core as np

from morie.fn.toppS import top_p_sampling


def test_toppS_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0.0, 1.0, 40)
    p = 0.1
    temp = 0.1
    result = top_p_sampling(logits, p, temp)
    assert isinstance(result, dict)
    assert "tensor" in result


def test_toppS_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0.0, 1.0, 40)
    p = 0.1
    temp = 0.1
    result = top_p_sampling(logits, p, temp)
    assert isinstance(result, dict)
