"""Tests for kmspn.kamath_t5_span_corruption."""

from morie.fn import _array_core as np

from morie.fn.kmspn import kamath_t5_span_corruption


def test_kmspn_basic():
    """Test basic functionality."""
    tokens = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kamath_t5_span_corruption(tokens)
    assert isinstance(result, dict)
    assert "estimate" in result or "input" in result


def test_kmspn_edge():
    """Test edge cases."""
    tokens = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kamath_t5_span_corruption(tokens)
    assert isinstance(result, dict)
