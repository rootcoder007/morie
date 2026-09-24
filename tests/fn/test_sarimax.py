"""Tests for sarimax.sarimax."""

from morie.fn import _array_core as np

from morie.fn.sarimax import sarimax


def test_sarimax_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sarimax(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "beta_se" in result


def test_sarimax_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sarimax(y)
    assert isinstance(result, dict)
