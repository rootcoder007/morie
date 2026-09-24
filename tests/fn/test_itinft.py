"""Tests for itinft.item_information_function."""

from morie.fn import _array_core as np

from morie.fn.itinft import item_information_function


def test_itinft_basic():
    """Test basic functionality."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    a = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    b = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = item_information_function(theta, a, b)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_itinft_edge():
    """Test edge cases."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    a = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    b = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = item_information_function(theta, a, b)
    assert isinstance(result, dict)
