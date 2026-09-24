"""Tests for sdxlcd.sdxl_unet."""

from morie.fn import _array_core as np

from morie.fn.sdxlcd import sdxl_unet


def test_sdxlcd_basic():
    """Test basic functionality."""
    h_original = 0.5
    w_original = 0.5
    result = sdxl_unet(h_original, w_original)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_sdxlcd_edge():
    """Test edge cases."""
    h_original = 0.5
    w_original = 0.5
    result = sdxl_unet(h_original, w_original)
    assert isinstance(result, dict)
