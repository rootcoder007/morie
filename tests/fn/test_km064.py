"""Tests for km064.kamath_ch4_loftq_objective."""

from morie.fn import _array_core as np

from morie.fn.km064 import kamath_ch4_loftq_objective


def test_km064_basic():
    """Test basic functionality."""
    W = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Q = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    A = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    B = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = kamath_ch4_loftq_objective(W, Q, A, B)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_km064_edge():
    """Test edge cases."""
    W = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Q = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    A = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    B = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = kamath_ch4_loftq_objective(W, Q, A, B)
    assert isinstance(result, dict)
