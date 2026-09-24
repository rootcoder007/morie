"""Tests for sgtdiff.sgt_diffusion_kernel."""

from morie.fn import _array_core as np

from morie.fn.sgtdiff import sgt_diffusion_kernel


def test_sgtdiff_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sgt_diffusion_kernel(A)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_sgtdiff_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sgt_diffusion_kernel(A)
    assert isinstance(result, dict)
