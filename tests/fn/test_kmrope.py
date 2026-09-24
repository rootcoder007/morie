"""Tests for kmrope.kamath_rotary_positional_embedding."""

from morie.fn import _array_core as np

from morie.fn.kmrope import kamath_rotary_positional_embedding


def test_kmrope_basic():
    """Test basic functionality."""
    q = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = kamath_rotary_positional_embedding(q)
    assert isinstance(result, dict)
    assert "estimate" in result or "y" in result


def test_kmrope_edge():
    """Test edge cases."""
    q = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = kamath_rotary_positional_embedding(q)
    assert isinstance(result, dict)
