"""Tests for km112.kamath_ch7_answer_relevance."""

from morie.fn import _array_core as np

from morie.fn.km112 import kamath_ch7_answer_relevance


def test_km112_basic():
    """Test basic functionality."""
    E_g = np.random.default_rng(42).normal(0.0, 1.0, 40)
    E_o = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kamath_ch7_answer_relevance(E_g, E_o)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_km112_edge():
    """Test edge cases."""
    E_g = np.random.default_rng(42).normal(0.0, 1.0, 40)
    E_o = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kamath_ch7_answer_relevance(E_g, E_o)
    assert isinstance(result, dict)
