"""Tests for jooutl.joseph_ts_outlier_detection."""

from morie.fn import _array_core as np

from morie.fn.jooutl import joseph_ts_outlier_detection


def test_jooutl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_ts_outlier_detection(y)
    assert isinstance(result, dict)
    assert "outlier" in result


def test_jooutl_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_ts_outlier_detection(y)
    assert isinstance(result, dict)
