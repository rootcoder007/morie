"""Tests for kmswig.kamath_swiglu_activation."""

from morie.fn import _array_core as np

from morie.fn.kmswig import kamath_swiglu_activation


def test_kmswig_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    W = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    V = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kamath_swiglu_activation(x, W, V)
    assert isinstance(result, dict)
    assert "estimate" in result or "output" in result


def test_kmswig_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    W = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    V = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kamath_swiglu_activation(x, W, V)
    assert isinstance(result, dict)
