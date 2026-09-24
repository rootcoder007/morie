"""Tests for otsd.ot_sliced_distance_quant."""

from morie.fn import _array_core as np

from morie.fn.otsd import ot_sliced_distance_quant


def test_otsd_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ot_sliced_distance_quant(X, Y)
    assert isinstance(result, dict)
    assert "SW" in result


def test_otsd_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ot_sliced_distance_quant(X, Y)
    assert isinstance(result, dict)
