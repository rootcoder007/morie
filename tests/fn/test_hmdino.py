"""Tests for hmdino.geron_dino."""

from morie.fn import _array_core as np

from morie.fn.hmdino import geron_dino


def test_hmdino_basic():
    """Test basic functionality."""
    images = np.random.default_rng(42).normal(0.0, 1.0, 40)
    student = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    teacher = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_dino(images, student, teacher)
    assert isinstance(result, dict)
    assert "estimate" in result or "loss" in result


def test_hmdino_edge():
    """Test edge cases."""
    images = np.random.default_rng(42).normal(0.0, 1.0, 40)
    student = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    teacher = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_dino(images, student, teacher)
    assert isinstance(result, dict)
