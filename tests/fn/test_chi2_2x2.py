"""Tests for chi2_2x2.chi2_2x2."""

from morie.fn import _array_core as np

from morie.fn.chi2_2x2 import chi2_2x2


def test_ca9e4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chi2_2x2(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca9e4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chi2_2x2(x)
    assert isinstance(result, dict)
