"""Tests for mareml.ma_random_reml."""

from morie.fn import _array_core as np

from morie.fn.mareml import ma_random_reml


def test_mareml_basic():
    """Test basic functionality."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    vi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_random_reml(yi, vi)
    assert isinstance(result, dict)
    assert "tau2" in result


def test_mareml_edge():
    """Test edge cases."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    vi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_random_reml(yi, vi)
    assert isinstance(result, dict)
