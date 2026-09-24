"""Tests for tukrr.tukey_regression."""

from morie.fn import _array_core as np

from morie.fn.tukrr import tukey_regression


def test_tukrr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = tukey_regression(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tukrr_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = tukey_regression(X, y)
    assert isinstance(result, dict)
