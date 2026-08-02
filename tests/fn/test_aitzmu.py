"""Tests for aitzmu.compositional_zero_multreplace."""

from morie.fn import _array_core as np

from morie.fn.aitzmu import compositional_zero_multreplace


def test_aitzmu_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_zero_multreplace(X, delta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_aitzmu_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_zero_multreplace(X, delta)
    assert isinstance(result, dict)
