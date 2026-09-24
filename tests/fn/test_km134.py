"""Tests for km134.kamath_ch9_clip_text_to_image."""

from morie.fn import _array_core as np

from morie.fn.km134 import kamath_ch9_clip_text_to_image


def test_km134_basic():
    """Test basic functionality."""
    L = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    V = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    sigma = 0.1
    result = kamath_ch9_clip_text_to_image(L, V, sigma)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_km134_edge():
    """Test edge cases."""
    L = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    V = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    sigma = 0.1
    result = kamath_ch9_clip_text_to_image(L, V, sigma)
    assert isinstance(result, dict)
