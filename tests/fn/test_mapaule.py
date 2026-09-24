"""Tests for mapaule.ma_paule_mandel."""

from morie.fn import _array_core as np

from morie.fn.mapaule import ma_paule_mandel


def test_mapaule_basic():
    """Test basic functionality."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    vi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_paule_mandel(yi, vi)
    assert isinstance(result, dict)
    assert "tau2" in result


def test_mapaule_edge():
    """Test edge cases."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    vi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_paule_mandel(yi, vi)
    assert isinstance(result, dict)
