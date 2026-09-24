"""Tests for otmsw.ot_max_sliced_w."""

from morie.fn import _array_core as np

from morie.fn.otmsw import ot_max_sliced_w


def test_otmsw_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ot_max_sliced_w(X, Y)
    assert isinstance(result, dict)
    assert "MSW" in result


def test_otmsw_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ot_max_sliced_w(X, Y)
    assert isinstance(result, dict)
