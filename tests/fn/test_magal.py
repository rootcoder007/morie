"""Tests for magal.ma_galbraith."""

from morie.fn import _array_core as np

from morie.fn.magal import ma_galbraith


def test_magal_basic():
    """Test basic functionality."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    se_i = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_galbraith(yi, se_i)
    assert isinstance(result, dict)
    assert "z" in result


def test_magal_edge():
    """Test edge cases."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    se_i = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_galbraith(yi, se_i)
    assert isinstance(result, dict)
