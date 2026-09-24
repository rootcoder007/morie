"""Tests for sensSlp.sen_slope."""

from morie.fn import _array_core as np

from morie.fn.sensSlp import sen_slope


def test_sensSlp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = sen_slope(y)
    assert isinstance(result, dict)
    assert "slope" in result


def test_sensSlp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = sen_slope(y)
    assert isinstance(result, dict)
