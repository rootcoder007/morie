"""Tests for rklfunc.ripley_l."""

from morie.fn import _array_core as np

from morie.fn.rklfunc import ripley_l


def test_rklfunc_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ripley_l(coords)
    assert isinstance(result, dict)
    assert "r" in result


def test_rklfunc_edge():
    """Test edge cases."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ripley_l(coords)
    assert isinstance(result, dict)
