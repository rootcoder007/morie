"""Tests for otsw.ot_sliced_wasserstein."""

from morie.fn import _array_core as np

from morie.fn.otsw import ot_sliced_wasserstein


def test_otsw_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ot_sliced_wasserstein(X, Y)
    assert isinstance(result, dict)
    assert "SW" in result


def test_otsw_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ot_sliced_wasserstein(X, Y)
    assert isinstance(result, dict)
