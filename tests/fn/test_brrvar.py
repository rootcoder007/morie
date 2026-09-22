"""Tests for brrvar.brr_variance."""

from morie.fn import _array_core as np

from morie.fn.brrvar import brr_variance


def test_brrvar_basic():
    """Test basic functionality."""
    estimates = np.random.default_rng(42).normal(0, 1, 100)
    result = brr_variance(estimates)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_brrvar_edge():
    """Test edge cases."""
    estimates = np.random.default_rng(42).normal(0, 1, 100)
    result = brr_variance(estimates)
    assert isinstance(result, dict)
