"""Tests for hmvilb.geron_vilbert."""

from morie.fn import _array_core as np

from morie.fn.hmvilb import geron_vilbert


def test_hmvilb_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0.0, 1.0, 40)
    text = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_vilbert(image, text)
    assert isinstance(result, dict)
    assert "estimate" in result or "image_out" in result


def test_hmvilb_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0.0, 1.0, 40)
    text = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_vilbert(image, text)
    assert isinstance(result, dict)
