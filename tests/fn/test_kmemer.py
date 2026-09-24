"""Tests for kmemer.kamath_emergent_abilities."""

from morie.fn import _array_core as np

from morie.fn.kmemer import kamath_emergent_abilities


def test_kmemer_basic():
    """Test basic functionality."""
    scales = np.random.default_rng(42).normal(0.0, 1.0, 40)
    scores = np.random.default_rng(42).normal(0.0, 1.0, 40)
    threshold = 0.1
    result = kamath_emergent_abilities(scales, scores, threshold)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_kmemer_edge():
    """Test edge cases."""
    scales = np.random.default_rng(42).normal(0.0, 1.0, 40)
    scores = np.random.default_rng(42).normal(0.0, 1.0, 40)
    threshold = 0.1
    result = kamath_emergent_abilities(scales, scores, threshold)
    assert isinstance(result, dict)
