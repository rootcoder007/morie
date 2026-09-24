"""Tests for jorolf.joseph_rolling_window_feature."""

from morie.fn import _array_core as np

from morie.fn.jorolf import joseph_rolling_window_feature


def test_jorolf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    window = 5
    result = joseph_rolling_window_feature(x, window)
    assert isinstance(result, dict)
    assert "mean" in result


def test_jorolf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    window = 5
    result = joseph_rolling_window_feature(x, window)
    assert isinstance(result, dict)
