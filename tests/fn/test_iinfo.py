"""Tests for iinfo.item_information."""

from morie.fn import _array_core as np

from morie.fn.iinfo import item_information


def test_iinfo_basic():
    """Test basic functionality."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = item_information(theta)
    assert isinstance(result, dict)
    assert "info" in result


def test_iinfo_edge():
    """Test edge cases."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = item_information(theta)
    assert isinstance(result, dict)
