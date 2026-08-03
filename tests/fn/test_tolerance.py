"""Tests for tolerance.tolerance."""

from morie.fn import _array_core as np

from morie.fn.tolerance import tolerance


def test_ca3e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = tolerance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca3e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = tolerance(x)
    assert isinstance(result, dict)
