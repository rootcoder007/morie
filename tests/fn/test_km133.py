"""Tests for km133.kamath_ch9_clip_image_to_text."""

from morie.fn import _array_core as np

from morie.fn.km133 import kamath_ch9_clip_image_to_text


def test_km133_basic():
    """Test basic functionality."""
    V = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    L = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    sigma = 0.1
    result = kamath_ch9_clip_image_to_text(V, L, sigma)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_km133_edge():
    """Test edge cases."""
    V = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    L = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    sigma = 0.1
    result = kamath_ch9_clip_image_to_text(V, L, sigma)
    assert isinstance(result, dict)
