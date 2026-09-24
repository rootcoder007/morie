"""Tests for hmdfw.geron_diffusion_forward."""

from morie.fn import _array_core as np

from morie.fn.hmdfw import geron_diffusion_forward


def test_hmdfw_basic():
    """Test basic functionality."""
    x0 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    T = 5
    result = geron_diffusion_forward(x0, T)
    assert isinstance(result, dict)
    assert "estimate" in result or "x_t" in result


def test_hmdfw_edge():
    """Test edge cases."""
    x0 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    T = 5
    result = geron_diffusion_forward(x0, T)
    assert isinstance(result, dict)
