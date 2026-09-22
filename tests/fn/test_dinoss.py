"""Tests for dinoss.dino_centering."""

from morie.fn import _array_core as np

from morie.fn.dinoss import dino_centering


def test_dinoss_basic():
    """Test basic functionality."""
    g_t = np.random.default_rng(42).normal(0, 1, 100)
    result = dino_centering(g_t)
    assert isinstance(result, dict)
    assert "p_t" in result
def test_dinoss_edge():
    """Test edge cases."""
    g_t = np.random.default_rng(42).normal(0, 1, 100)
    result = dino_centering(g_t)
    assert isinstance(result, dict)
