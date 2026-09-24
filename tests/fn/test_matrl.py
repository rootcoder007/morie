"""Tests for matrl.ma_three_level."""

from morie.fn import _array_core as np

from morie.fn.matrl import ma_three_level


def test_matrl_basic():
    """Test basic functionality."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    vi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    cluster = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_three_level(yi, vi, cluster)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_matrl_edge():
    """Test edge cases."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    vi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    cluster = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_three_level(yi, vi, cluster)
    assert isinstance(result, dict)
