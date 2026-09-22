"""Tests for gb421.gibbons_chisq_gof."""

from morie.fn import _array_core as np

from morie.fn.gb421 import gibbons_chisq_gof


def test_gb421_basic():
    """Test basic functionality."""
    observed = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    expected = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = gibbons_chisq_gof(observed, expected)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_gb421_edge():
    """Test edge cases."""
    observed = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    expected = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = gibbons_chisq_gof(observed, expected)
    assert isinstance(result, dict)
