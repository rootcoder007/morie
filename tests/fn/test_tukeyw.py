"""Tests for tukeyw.tukey_biweight."""

from morie.fn import _array_core as np

from morie.fn.tukeyw import tukey_biweight


def test_tukeyw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tukey_biweight(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tukeyw_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tukey_biweight(y)
    assert isinstance(result, dict)
