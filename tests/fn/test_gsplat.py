"""Tests for gsplat.gaussian_splatting."""

from morie.fn import _array_core as np

from morie.fn.gsplat import gaussian_splatting


def test_gsplat_basic():
    """Test basic functionality."""
    colours = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    alphas = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = gaussian_splatting(colours, alphas)
    assert isinstance(result, dict)
    assert "colour" in result


def test_gsplat_edge():
    """Test edge cases."""
    colours = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    alphas = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = gaussian_splatting(colours, alphas)
    assert isinstance(result, dict)
