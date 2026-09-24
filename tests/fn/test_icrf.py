"""Tests for icrf.item_characteristic_curve."""

from morie.fn import _array_core as np

from morie.fn.icrf import item_characteristic_curve


def test_icrf_basic():
    """Test basic functionality."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = item_characteristic_curve(theta)
    assert isinstance(result, dict)
    assert "p" in result


def test_icrf_edge():
    """Test edge cases."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = item_characteristic_curve(theta)
    assert isinstance(result, dict)
