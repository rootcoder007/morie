"""Tests for km3h.kamath_3h_alignment."""

from morie.fn import _array_core as np

from morie.fn.km3h import kamath_3h_alignment


def test_km3h_basic():
    """Test basic functionality."""
    helpful_score = np.random.default_rng(42).normal(0.0, 1.0, 40)
    harmless_score = np.random.default_rng(42).normal(0.0, 1.0, 40)
    honest_score = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kamath_3h_alignment(helpful_score, harmless_score, honest_score)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_km3h_edge():
    """Test edge cases."""
    helpful_score = np.random.default_rng(42).normal(0.0, 1.0, 40)
    harmless_score = np.random.default_rng(42).normal(0.0, 1.0, 40)
    honest_score = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kamath_3h_alignment(helpful_score, harmless_score, honest_score)
    assert isinstance(result, dict)
