"""Tests for weisL.wl_kernel."""

from morie.fn import _array_core as np

from morie.fn.weisL import wl_kernel


def test_weisL_basic():
    """Test basic functionality."""
    G1 = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    G2 = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = wl_kernel(G1, G2)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_weisL_edge():
    """Test edge cases."""
    G1 = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    G2 = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = wl_kernel(G1, G2)
    assert isinstance(result, dict)
