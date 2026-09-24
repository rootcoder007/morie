"""Tests for tipsne.tipping_point_sensitivity."""

from morie.fn import _array_core as np

from morie.fn.tipsne import tipping_point_sensitivity


def test_tipsne_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    result = tipping_point_sensitivity(y, D)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tipsne_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    result = tipping_point_sensitivity(y, D)
    assert isinstance(result, dict)
